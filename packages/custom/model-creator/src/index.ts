/**
 * A Cordis plugin that registers the `/model-creator` slash command: given an
 * n8n webhook URL, it builds a brand-new DSH custom-model bridge package for
 * that workflow end to end — match the URL to its workflow and read its
 * definition via n8n's own REST API, scaffold a
 * `packages/custom/<derived-name>/` package from the standard bridge
 * template, install and build it, register it in the profile's
 * `cordis.patch.yml` and `settings.yaml`, verify the generated bridge logic
 * directly against the live webhook, and report back what is left for a
 * person to finish by hand.
 *
 * Everything above is plain, deterministic code — the same shape as
 * `permaculture-command`. The one place this command cannot be deterministic
 * is reading how a given workflow's Respond node actually shapes its answer:
 * different workflows reply differently, and no fixed rule reads all of
 * them. For that one step, this plugin delegates to a one-shot subagent
 * (`@deepseek-ai/dsh-subagent`) with a small, strict output schema, and
 * splices its answer into the otherwise-fixed generation logic below.
 *
 * This command cannot restart the DSH process it runs inside of — doing so
 * would kill it mid-execution — so it never attempts to. It verifies the
 * generated bridge's own fetch-and-parse logic directly against the live n8n
 * webhook (bypassing DSH's own HTTP route entirely), then hands the human
 * the two steps only a running DSH process can finish: entering the
 * generated shared secret into Settings > Models, and restarting so the new
 * provider and package are picked up.
 *
 * @module @deepseek-ai/dsh-model-creator
 */

import { randomBytes } from 'node:crypto'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { Context, Service } from '@deepseek-ai/cordis'
import type { Agent } from '@deepseek-ai/dsh-agent'
import { boundContextSummary, createUserMessage } from '@deepseek-ai/dsh-llm'
import type {} from '@deepseek-ai/dsh-commands'
import type {} from '@deepseek-ai/dsh-host-webserver'
import type { SubagentResult } from '@deepseek-ai/dsh-subagent'
import type { ObjectJsonSchema } from '@deepseek-ai/dsh-tools'
import { parseDocument as parseYamlDocument } from 'yaml'

const execFileAsync = promisify(execFile)

declare module '@deepseek-ai/cordis' {
  interface Context {
    modelCreator: ModelCreatorController
  }
}

/** Deployment-owned config: which n8n instance this command may read workflows from. */
export interface ModelCreatorConfig {
  /** Base URL of the n8n instance, e.g. `https://your-n8n.example.com` — no trailing slash. */
  n8nBaseUrl: string
  /** n8n API key sent as `X-N8N-API-KEY` to read workflow definitions via n8n's REST API. */
  n8nApiKey: string
  /** Absolute path to the repo root (the `deepseek-harness` checkout), for running pnpm and writing packages. */
  repoRoot: string
  /** Absolute path to the DSH profile directory holding `cordis.patch.yml`. */
  profileDir: string
  /** Absolute path to the DSH home directory (e.g. `~/.dsh`) holding the shared, cross-profile `settings.yaml`. */
  dshHomeDir: string
}

/** Validate deployment-owned config at load time — fail loudly here, not silently at request time. */
function resolveConfig(config: unknown): ModelCreatorConfig {
  if (typeof config !== 'object' || config === null) {
    throw new Error('ModelCreatorConfig must be an object')
  }
  const record = config as Record<string, unknown>
  const n8nBaseUrl = record.n8nBaseUrl
  if (typeof n8nBaseUrl !== 'string' || n8nBaseUrl.trim() === '') {
    throw new Error('ModelCreatorConfig needs a non-empty string `n8nBaseUrl`')
  }
  const n8nApiKey = record.n8nApiKey
  if (typeof n8nApiKey !== 'string' || n8nApiKey.trim() === '') {
    throw new Error('ModelCreatorConfig needs a non-empty string `n8nApiKey`')
  }
  const repoRoot = record.repoRoot
  if (typeof repoRoot !== 'string' || repoRoot.trim() === '') {
    throw new Error('ModelCreatorConfig needs a non-empty string `repoRoot`')
  }
  const profileDir = record.profileDir
  if (typeof profileDir !== 'string' || profileDir.trim() === '') {
    throw new Error('ModelCreatorConfig needs a non-empty string `profileDir`')
  }
  const dshHomeDir = record.dshHomeDir
  if (typeof dshHomeDir !== 'string' || dshHomeDir.trim() === '') {
    throw new Error('ModelCreatorConfig needs a non-empty string `dshHomeDir`')
  }
  const unknownKeys = Object.keys(record).filter(
    key =>
      key !== 'n8nBaseUrl' &&
      key !== 'n8nApiKey' &&
      key !== 'repoRoot' &&
      key !== 'profileDir' &&
      key !== 'dshHomeDir',
  )
  if (unknownKeys.length > 0) {
    throw new Error(
      `ModelCreatorConfig has unknown key(s) ${unknownKeys.join(', ')}`,
    )
  }
  return {
    n8nBaseUrl: n8nBaseUrl.replace(/\/+$/, ''),
    n8nApiKey,
    repoRoot,
    profileDir,
    dshHomeDir,
  }
}

// ---------------------------------------------------------------------------
// Phase 2 — resolve the n8n workflow from its webhook URL (deterministic)
// ---------------------------------------------------------------------------

/** What Phase 2 needs from the workflow before anything can be generated. */
interface ResolvedWorkflow {
  readonly workflowId: string
  readonly title: string
  readonly webhookUrl: string
  readonly active: boolean
  readonly headerAuthAttached: boolean
  /** The trigger + terminal node's raw JSON, handed to the subagent in Phase 3 — nothing else. */
  readonly relevantNodes: unknown
}

type WorkflowResolution =
  | { kind: 'resolved'; workflow: ResolvedWorkflow }
  | { kind: 'error'; text: string }

/** What parsing the given webhook URL against the configured n8n instance can produce. */
type WebhookReference =
  | { kind: 'path'; path: string }
  | { kind: 'error'; text: string }

/**
 * Parse the webhook URL /model-creator was given into the path segment n8n's
 * workflow list can be matched against, rejecting the two most common
 * mistakes early: a URL from a different n8n instance, and n8n's Test URL
 * (`/webhook-test/...`), which only responds while the editor is actively
 * listening and will never resolve to anything useful here.
 */
function parseWebhookReference(n8nBaseUrl: string, reference: string): WebhookReference {
  const trimmed = reference.trim()
  if (!trimmed.startsWith(n8nBaseUrl)) {
    return {
      kind: 'error',
      text: `"${trimmed}" doesn't look like a webhook URL from your n8n instance at ${n8nBaseUrl} — check you copied the right URL.`,
    }
  }
  if (/\/webhook-test\/[A-Za-z0-9_-]+/.exec(trimmed)) {
    return {
      kind: 'error',
      text: 'That\'s the Test URL — it only responds while you\'re watching the workflow in the n8n editor. Open the workflow, publish it (Active), and copy the Production URL instead (same node, without "-test").',
    }
  }
  const pathMatch = /\/webhook\/([A-Za-z0-9_-]+)/.exec(trimmed)
  if (pathMatch?.[1] === undefined) {
    return {
      kind: 'error',
      text: `Could not find a webhook path in "${trimmed}" — pass the workflow's Production webhook URL.`,
    }
  }
  return { kind: 'path', path: pathMatch[1] }
}

/** One page of n8n's `GET /api/v1/workflows` list endpoint. */
interface N8nWorkflowListPage {
  data: Record<string, unknown>[]
  nextCursor: string | null
}

/** Fetch one page of the workflow list, optionally continuing from a cursor. */
async function fetchWorkflowListPage(
  config: ModelCreatorConfig,
  cursor: string | undefined,
): Promise<N8nWorkflowListPage> {
  const url = new URL(`${config.n8nBaseUrl}/api/v1/workflows`)
  if (cursor !== undefined) url.searchParams.set('cursor', cursor)
  const response = await fetch(url, {
    headers: { 'X-N8N-API-KEY': config.n8nApiKey },
  })
  if (!response.ok) {
    throw new Error(`n8n returned HTTP ${response.status} listing workflows — check the API key can read them.`)
  }
  const body = (await response.json()) as { data?: unknown; nextCursor?: unknown }
  const data = Array.isArray(body.data) ? (body.data as Record<string, unknown>[]) : []
  const nextCursor = typeof body.nextCursor === 'string' ? body.nextCursor : null
  return { data, nextCursor }
}

/** List every workflow on the configured n8n instance, following pagination. */
async function listAllWorkflows(config: ModelCreatorConfig): Promise<Record<string, unknown>[]> {
  const all: Record<string, unknown>[] = []
  let cursor: string | undefined
  do {
    const page = await fetchWorkflowListPage(config, cursor)
    all.push(...page.data)
    cursor = page.nextCursor ?? undefined
  } while (cursor !== undefined)
  return all
}

/** A workflow whose Webhook Trigger node's configured path matches the one being resolved. */
interface WebhookPathMatch {
  workflow: Record<string, unknown>
  triggerNode: Record<string, unknown>
  nodes: Record<string, unknown>[]
}

/** Find every workflow on the instance with a Webhook Trigger node at the given path. */
function findWorkflowsByWebhookPath(
  workflows: Record<string, unknown>[],
  webhookPath: string,
): WebhookPathMatch[] {
  const matches: WebhookPathMatch[] = []
  for (const workflow of workflows) {
    const nodes = Array.isArray(workflow.nodes) ? (workflow.nodes as Record<string, unknown>[]) : []
    const triggerNode = nodes.find((node) => {
      if (typeof node.type !== 'string' || !node.type.includes('webhook')) return false
      const params = (node.parameters ?? {}) as Record<string, unknown>
      return params.path === webhookPath
    })
    if (triggerNode !== undefined) matches.push({ workflow, triggerNode, nodes })
  }
  return matches
}

/**
 * Fetch every workflow from n8n's REST API and pull out what's needed to
 * build a bridge for the one whose Webhook Trigger path matches the given
 * URL: its title, its published state, and whether Header Auth is attached
 * (existence only — the actual secret value is never exposed by this API,
 * and this command doesn't support Header Auth workflows at all).
 *
 * A bare webhook URL only carries n8n's webhook *path*, not the workflow's
 * internal id the older by-id lookup used, and n8n's API has no way to look
 * a workflow up by webhook path directly — so this lists every workflow on
 * the instance and scans their trigger nodes for the matching path.
 */
async function resolveN8nWorkflow(config: ModelCreatorConfig, reference: string): Promise<WorkflowResolution> {
  const parsed = parseWebhookReference(config.n8nBaseUrl, reference)
  if (parsed.kind === 'error') return { kind: 'error', text: parsed.text }

  let workflows: Record<string, unknown>[]
  try {
    workflows = await listAllWorkflows(config)
  } catch (error) {
    return { kind: 'error', text: error instanceof Error ? error.message : String(error) }
  }

  const matches = findWorkflowsByWebhookPath(workflows, parsed.path)
  if (matches.length === 0) {
    return {
      kind: 'error',
      text: `No workflow on your n8n instance has a webhook at "/${parsed.path}" — check the URL, or make sure the workflow's been imported and saved.`,
    }
  }
  const activeMatches = matches.filter(match => match.workflow.active === true)
  if (activeMatches.length > 1) {
    const titles = activeMatches
      .map(match => (typeof match.workflow.name === 'string' ? match.workflow.name : String(match.workflow.id)))
      .join('", "')
    return {
      kind: 'error',
      text: `Multiple active workflows use the webhook path "/${parsed.path}" ("${titles}") — n8n shouldn't normally allow this; rename one of their trigger paths before continuing.`,
    }
  }
  const match = activeMatches[0] ?? matches[0]
  if (match === undefined) {
    return { kind: 'error', text: 'Unexpected: failed to resolve a matching workflow.' }
  }

  const triggerParams = (match.triggerNode.parameters ?? {}) as Record<string, unknown>
  const headerAuthAttached = triggerParams.authentication === 'headerAuth'
  const title = typeof match.workflow.name === 'string' ? match.workflow.name : parsed.path
  const active = match.workflow.active === true
  const workflowId = typeof match.workflow.id === 'string' ? match.workflow.id : parsed.path

  // Only the trigger and the terminal (last) node go to the subagent in
  // Phase 3 — not the whole workflow, which can contain unrelated internal
  // logic the response-shape question has no need to see.
  const terminalNode = match.nodes.length > 0 ? match.nodes[match.nodes.length - 1] : undefined
  const relevantNodes = { trigger: match.triggerNode, terminal: terminalNode }

  return {
    kind: 'resolved',
    workflow: {
      workflowId,
      title,
      webhookUrl: reference.trim(),
      active,
      headerAuthAttached,
      relevantNodes,
    },
  }
}

/** Turn a workflow's display title into a short, kebab-case package/model name. */
function deriveModelName(title: string): string {
  const slug = title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 40)
    .replace(/-+$/g, '')
  return slug === '' ? 'custom-model' : slug
}

// ---------------------------------------------------------------------------
// Phase 3 — hand the one unpredictable part to a subagent
// ---------------------------------------------------------------------------

/** The one thing Phase 3 asks the subagent to decide. */
interface ResponseShapePlan {
  /** Whether the workflow already replies as a bare string or `{ "output": "..." }` — the template's default. */
  matchesStandardShape: boolean
  /**
   * When `matchesStandardShape` is false: a short, self-contained TypeScript
   * expression body for a function `(raw: string) => string` that extracts
   * the answer text from the workflow's actual raw response body. Absent
   * when the standard shape already applies.
   */
  customParseBody?: string
  /** One sentence explaining what the workflow returns, for the final report. */
  explanation: string
}

const RESPONSE_SHAPE_SCHEMA: ObjectJsonSchema = {
  type: 'object',
  additionalProperties: false,
  required: ['matchesStandardShape', 'explanation'],
  properties: {
    matchesStandardShape: { type: 'boolean' },
    customParseBody: { type: 'string' },
    explanation: { type: 'string' },
  },
}

/**
 * Ask a one-shot subagent to read the workflow's trigger and terminal node
 * and decide whether its reply already matches DSH's standard bridge
 * convention (a bare string or `{ "output": "..." }`), or hand back the
 * small parsing adjustment needed when it doesn't. This is the one
 * judgment-based step in the whole command; everything before and after is
 * fixed code.
 */
async function planResponseShape(
  ctx: Context,
  agent: Agent,
  workflow: ResolvedWorkflow,
): Promise<ResponseShapePlan> {
  const prompt = [
    'A DSH custom-model bridge expects an n8n webhook to reply either as a bare JSON string, ',
    'or as `{ "output": "<answer text>" }`. Given the trigger and terminal node below from a ',
    'real n8n workflow, decide whether its actual reply already matches that shape.\n\n',
    `Nodes:\n${JSON.stringify(workflow.relevantNodes, null, 2)}\n\n`,
    'If it matches, set matchesStandardShape to true and omit customParseBody. If it does not, ',
    'set matchesStandardShape to false and give a customParseBody: the body of a TypeScript ',
    'function `(raw: string) => string` (just the statements, no signature) that extracts the ',
    "answer text from the workflow's actual raw HTTP response body. Keep it defensive: fall back ",
    'to the raw text if parsing fails, and never throw.',
  ].join('')

  const controller = new AbortController()
  const run = await ctx.subagents.start('spawn', {
    prompt: [{ type: 'text', text: prompt }],
    parent: agent,
    signal: controller.signal,
    outputSchema: RESPONSE_SHAPE_SCHEMA,
  })
  let result: SubagentResult
  try {
    result = await run.result
  } finally {
    await run.dispose()
  }

  if (result.stopReason !== 'completed' || result.structured === undefined) {
    // Fail toward the safe default rather than blocking the whole command on
    // a subagent hiccup — the standard template already handles a bare
    // string or `{ output }`, which covers most workflows.
    return {
      matchesStandardShape: true,
      explanation: `Could not get a structured answer from the response-shape check (${result.stopReason}); assuming the standard shape.`,
    }
  }
  return result.structured as ResponseShapePlan
}

// ---------------------------------------------------------------------------
// Phase 4 — scaffold the new bridge package
// ---------------------------------------------------------------------------

/** Everything Phase 4 needs to render a complete bridge package. */
interface BridgePlan {
  packageName: string
  modelId: string
  webhookUrl: string
  apiKey: string
  responseShape: ResponseShapePlan
}

function renderBridgePackageJson(plan: BridgePlan): string {
  return `${JSON.stringify(
    {
      name: `@deepseek-ai/dsh-${plan.packageName}`,
      description: `Bridges DSH's model picker to the n8n automation for ${plan.packageName}. Generated by /model-creator.`,
      version: '0.0.1',
      private: true,
      type: 'module',
      main: 'lib/index.js',
      types: 'lib/types/index.d.ts',
      exports: {
        '.': { types: './lib/types/index.d.ts', default: './lib/index.js' },
        './package.json': './package.json',
      },
      license: 'MIT',
      peerDependencies: {
        '@deepseek-ai/dsh-host-webserver': 'workspace:^',
        '@deepseek-ai/dsh-credentials': 'workspace:^',
        '@deepseek-ai/cordis': 'workspace:^',
      },
      devDependencies: {
        '@deepseek-ai/dsh-host-webserver': 'workspace:^',
        '@deepseek-ai/dsh-credentials': 'workspace:^',
        '@deepseek-ai/cordis': 'workspace:^',
      },
    },
    null,
    2,
  )}\n`
}

function renderBridgeTsconfig(): string {
  return `${JSON.stringify(
    {
      extends: '../../../tsconfig.base.json',
      compilerOptions: { rootDir: 'src', outDir: 'lib/types' },
      include: ['src'],
      references: [
        { path: '../../../vendor/cosmokit' },
        { path: '../../../vendor/cordis' },
        { path: '../../host/webserver' },
        { path: '../../credentials/credentials' },
      ],
    },
    null,
    2,
  )}\n`
}

/**
 * Render the bridge's `src/index.ts`. This is the same template worked
 * through by hand in the model-setup guide (Steps 5-7): validate config,
 * call the webhook, parse its answer, wrap it as an SSE stream, register the
 * route. The one substituted piece is the parse function's body, which is
 * either the template's own default (bare string or `{ output }`) or the
 * subagent's `customParseBody` from Phase 3.
 */
function renderBridgeSource(plan: BridgePlan): string {
  const parseBody =
    plan.responseShape.customParseBody ??
    [
      'let answer = raw',
      'try {',
      '  const data: unknown = JSON.parse(raw)',
      "  if (typeof data === 'string') {",
      '    answer = data',
      "  } else if (data !== null && typeof data === 'object' && typeof (data as Record<string, unknown>).output === 'string') {",
      '    answer = (data as Record<string, unknown>).output as string',
      '  }',
      '} catch {',
      '  // Not JSON — fall back to the raw response body as-is.',
      '}',
      'return answer.trim()',
    ].join('\n')

  return `/**
 * Generated by /model-creator for the "${plan.packageName}" n8n automation.
 * ${plan.responseShape.explanation}
 * @module @deepseek-ai/dsh-${plan.packageName}
 */

import { randomUUID } from 'node:crypto'
import type { IncomingMessage, ServerResponse } from 'node:http'
import { Context, Service } from '@deepseek-ai/cordis'
import type {} from '@deepseek-ai/dsh-host-webserver'
// Pulls in this module's \`declare module '@deepseek-ai/cordis'\` augmentation
// too, which is what makes \`ctx.get('credentials')\` below typecheck.
import { credentialRef, isCredentialRefName } from '@deepseek-ai/dsh-credentials'

declare module '@deepseek-ai/cordis' {
  interface Context {
    ${plan.packageName.replace(/-([a-z])/g, (_m, c: string) => c.toUpperCase())}Bridge: BridgeController
  }
}

export interface BridgeConfig {
  webhookUrl: string
  apiKey: string
  modelId: string
  path?: string
  /**
   * Character budget for the session title this route derives for its own
   * detected title-generation requests (see \`SESSION_TITLE_REQUEST_PREFIX\`
   * below) instead of spending an n8n call on them.
   * @default 60
   */
  sessionTitleMaxChars?: number
  /**
   * How this route answers its own detected session-title requests: \`'llm'\`
   * (the default) asks a real model to summarize the request into a short
   * title, falling back to local truncation on any failure -- bad or
   * missing key, network error, timeout, or a malformed response -- so a
   * title is always produced. \`'local'\` always truncates instead, with no
   * network call. (Same mechanism already applied by hand in
   * permaculture-model-bridge; ported here so every /model-creator-generated
   * bridge gets it too.)
   * @default 'llm'
   */
  titleMode?: 'local' | 'llm'
  /**
   * Name of the credential reference (or env var) holding the API key sent
   * to \`titleLlmBaseUrl\`. Resolved through \`ctx.credentials\` first -- same
   * as a built-in provider's \`apiKeyEnv\`, so a key already entered for, say,
   * the \`openrouter\` provider is found here too -- then through a literal
   * \`process.env\` lookup. Only used when \`titleMode\` is \`'llm'\`.
   * @default 'OPENROUTER_API_KEY'
   */
  titleLlmApiKeyEnv?: string
  /** Model id sent to \`titleLlmBaseUrl\` for title generation. @default 'anthropic/claude-sonnet-5' */
  titleLlmModel?: string
  /** OpenAI Chat Completions-compatible endpoint called for title generation. @default 'https://openrouter.ai/api/v1/chat/completions' */
  titleLlmBaseUrl?: string
  /** Deadline for the title-generation request, in milliseconds, before falling back to local truncation. @default 15000 */
  titleLlmTimeoutMs?: number
}

interface ResolvedConfig {
  webhookUrl: string
  apiKey: string
  modelId: string
  path: string
  sessionTitleMaxChars: number
  titleMode: 'local' | 'llm'
  titleLlmApiKeyEnv: string
  titleLlmModel: string
  titleLlmBaseUrl: string
  titleLlmTimeoutMs: number
}

const DEFAULT_PATH = '/${plan.packageName}/v1/chat/completions'
const DEFAULT_SESSION_TITLE_MAX_CHARS = 60
const DEFAULT_TITLE_MODE: 'local' | 'llm' = 'llm'
const DEFAULT_TITLE_LLM_API_KEY_ENV = 'OPENROUTER_API_KEY'
const DEFAULT_TITLE_LLM_MODEL = 'anthropic/claude-sonnet-5'
const DEFAULT_TITLE_LLM_BASE_URL = 'https://openrouter.ai/api/v1/chat/completions'
const DEFAULT_TITLE_LLM_TIMEOUT_MS = 15000

function resolveConfig(config: unknown): ResolvedConfig {
  if (typeof config !== 'object' || config === null) {
    throw new Error('BridgeConfig must be an object')
  }
  const record = config as Record<string, unknown>
  const webhookUrl = record.webhookUrl
  if (typeof webhookUrl !== 'string' || webhookUrl.trim() === '') {
    throw new Error('BridgeConfig needs a non-empty string \`webhookUrl\`')
  }
  const apiKey = record.apiKey
  if (typeof apiKey !== 'string' || apiKey.trim() === '') {
    throw new Error('BridgeConfig needs a non-empty string \`apiKey\`')
  }
  const modelId = record.modelId
  if (typeof modelId !== 'string' || modelId.trim() === '') {
    throw new Error('BridgeConfig needs a non-empty string \`modelId\`')
  }
  const path = record.path
  const titleMode = record.titleMode === 'local' || record.titleMode === 'llm' ? record.titleMode : DEFAULT_TITLE_MODE
  return {
    webhookUrl,
    apiKey,
    modelId,
    path: (path as string | undefined) ?? DEFAULT_PATH,
    sessionTitleMaxChars: (record.sessionTitleMaxChars as number | undefined) ?? DEFAULT_SESSION_TITLE_MAX_CHARS,
    titleMode,
    titleLlmApiKeyEnv: (record.titleLlmApiKeyEnv as string | undefined) ?? DEFAULT_TITLE_LLM_API_KEY_ENV,
    titleLlmModel: (record.titleLlmModel as string | undefined) ?? DEFAULT_TITLE_LLM_MODEL,
    titleLlmBaseUrl: (record.titleLlmBaseUrl as string | undefined) ?? DEFAULT_TITLE_LLM_BASE_URL,
    titleLlmTimeoutMs: (record.titleLlmTimeoutMs as number | undefined) ?? DEFAULT_TITLE_LLM_TIMEOUT_MS,
  }
}

/** Extracts the answer text from the webhook's raw response body. */
function parseAnswer(raw: string): string {

${parseBody
  .split('\n')
  .map(line => `  ${line}`)
  .join('\n')}
}

async function fetchGroundedAnswer(
  config: ResolvedConfig,
  question: string,
  sessionId: string | undefined,
): Promise<string> {
  const response = await fetch(config.webhookUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      chatInput: question,
      ...sessionId === undefined ? {} : { sessionId },
    }),
  })
  if (!response.ok) throw new Error(\`n8n webhook returned HTTP \${response.status}\`)
  const raw = await response.text()
  const answer = parseAnswer(raw)
  if (answer === '') throw new Error('n8n webhook returned an empty response')
  return answer
}

function readRequestBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = []
    req.on('data', (chunk: Buffer) => chunks.push(chunk))
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')))
    req.on('error', reject)
  })
}

/**
 * Parse an OpenAI-style message 'content' field (plain string or
 * content-part array) into its text. Only text parts are used -- this
 * bridge sends only plain text to n8n.
 */
function extractTextContent(content: unknown): string | undefined {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return undefined

  const textParts: string[] = []
  for (const part of content) {
    if (part === null || typeof part !== 'object') continue
    const record = part as Record<string, unknown>
    if (record.type === 'text' && typeof record.text === 'string') textParts.push(record.text)
  }

  return textParts.length === 0 ? undefined : textParts.join('\\n')
}

/**
 * Prefixes DSH's own agent loop appends, as additional trailing \`role:
 * "user"\` messages, after the person's real turn -- a "workspace instructions"
 * reminder and a "current runtime context" snapshot. A naive "just take the
 * last user message" extraction picks up one of these instead of what the
 * person actually typed. DSH may add more such synthetic messages over
 * time; widen this list rather than assume the last \`user\` message is
 * always the real one. (Same fix already applied by hand in
 * permaculture-model-bridge; ported here so every /model-creator-generated
 * bridge gets it too.)
 */
const SYNTHETIC_CONTEXT_PREFIXES = ['<system-reminder>', 'Current runtime context.']

/** Whether an extracted user-turn text is one of DSH's own injected context messages rather than something the person typed. */
function isSyntheticContext(text: string): boolean {
  return SYNTHETIC_CONTEXT_PREFIXES.some(prefix => text.startsWith(prefix))
}

/**
 * Find the person's actual last message text, skipping synthetic context
 * messages.
 */
function extractLastUserText(messages: unknown): string | undefined {
  if (!Array.isArray(messages)) return undefined
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const m = messages[i] as Record<string, unknown>
    if (m?.role !== 'user') continue
    const text = extractTextContent(m.content)
    if (text === undefined || isSyntheticContext(text)) continue
    return text
  }
  return undefined
}

/**
 * The exact, stable prefix \`@deepseek-ai/dsh-session-title-llm\` frames its
 * one auxiliary request with. Recognizing this lets this route answer
 * session-title requests locally (or via a real model, per \`titleMode\`)
 * instead of forwarding them to n8n -- otherwise a live n8n run gets spent
 * summarizing nothing, and the answer comes back in the automation's own
 * voice instead of a short title. (Same fix already applied by hand in
 * permaculture-model-bridge; ported here so every /model-creator-generated
 * bridge gets it too.)
 */
const SESSION_TITLE_REQUEST_PREFIX = 'Generate the session title from this JSON array of human messages:\\n'

interface SessionTitleSourceMessage {
  readonly text?: unknown
}

/** Parse a detected session-title request and return its first selected message's text, whitespace-collapsed and trimmed. */
function extractSessionTitleSourceText(requestText: string): string | undefined {
  const json = requestText.slice(SESSION_TITLE_REQUEST_PREFIX.length)
  let parsed: unknown
  try {
    parsed = JSON.parse(json)
  } catch {
    return undefined
  }
  if (!Array.isArray(parsed) || parsed.length === 0) return undefined
  const first = parsed[0] as SessionTitleSourceMessage
  if (typeof first.text !== 'string') return undefined
  const collapsed = first.text.replace(/\\s+/g, ' ').trim()
  return collapsed === '' ? undefined : collapsed
}

/** Truncate \`text\` to \`maxChars\` on a word boundary, appending an ellipsis when cut. */
function truncateTitle(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text
  const truncated = text.slice(0, maxChars)
  const lastSpace = truncated.lastIndexOf(' ')
  return \`\${lastSpace > 0 ? truncated.slice(0, lastSpace) : truncated}…\`
}

/**
 * Derive a short title locally by truncating a detected session-title
 * request's first message. Also the fallback \`titleMode: 'llm'\` uses on
 * any failure.
 */
function deriveSessionTitle(requestText: string, maxChars: number): string | undefined {
  const collapsed = extractSessionTitleSourceText(requestText)
  return collapsed === undefined ? undefined : truncateTitle(collapsed, maxChars)
}

const TITLE_LLM_SYSTEM_PROMPT = 'Write a short session title (a few words, plain text, no quotes, no trailing punctuation) that summarizes the user message that follows. Reply with only the title -- no preamble, no explanation.'

/**
 * Resolve \`name\` to a secret value through \`ctx.credentials\` first,
 * falling back to a literal \`process.env\` lookup. Never throws -- a
 * resolution failure is just "not found" here.
 */
async function resolveTitleLlmApiKey(ctx: Context, name: string): Promise<string | undefined> {
  if (isCredentialRefName(name)) {
    const credentials = ctx.get('credentials')
    if (credentials !== undefined) {
      try {
        const resolved = await credentials.resolve(credentialRef(name))
        if (resolved !== undefined) return resolved.value
      } catch {
        // Fall through to the plain env-var check below.
      }
    }
  }
  return process.env[name]
}

/**
 * Ask a real model to summarize a detected session-title request into a
 * short title. Returns \`undefined\` on ANY failure, so the caller can fall
 * back to local truncation and still produce a title. Never throws.
 */
async function deriveSessionTitleViaLlm(ctx: Context, config: ResolvedConfig, requestText: string): Promise<string | undefined> {
  const apiKey = await resolveTitleLlmApiKey(ctx, config.titleLlmApiKeyEnv)
  if (!apiKey) return undefined

  const sourceText = extractSessionTitleSourceText(requestText)
  if (sourceText === undefined) return undefined

  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), config.titleLlmTimeoutMs)
  try {
    const response = await fetch(config.titleLlmBaseUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: \`Bearer \${apiKey}\` },
      body: JSON.stringify({
        model: config.titleLlmModel,
        messages: [
          { role: 'system', content: TITLE_LLM_SYSTEM_PROMPT },
          { role: 'user', content: sourceText },
        ],
        max_tokens: 30,
        temperature: 0.3,
      }),
      signal: controller.signal,
    })
    if (!response.ok) return undefined

    const data: unknown = await response.json().catch(() => undefined)
    if (data === null || typeof data !== 'object') return undefined
    const choices = (data as Record<string, unknown>).choices
    if (!Array.isArray(choices) || choices.length === 0) return undefined
    const first = choices[0]
    if (first === null || typeof first !== 'object') return undefined
    const message = (first as Record<string, unknown>).message
    if (message === null || typeof message !== 'object') return undefined
    const content = (message as Record<string, unknown>).content
    if (typeof content !== 'string') return undefined

    const collapsed = content.replace(/\\s+/g, ' ').trim().replace(/^["']|["']$/g, '')
    if (collapsed === '') return undefined
    return truncateTitle(collapsed, config.sessionTitleMaxChars)
  } catch {
    return undefined
  } finally {
    clearTimeout(timer)
  }
}

function writeSseAnswer(res: ServerResponse, modelId: string, answer: string): void {
  res.writeHead(200, { 'Content-Type': 'text/event-stream; charset=utf-8', 'Cache-Control': 'no-cache', Connection: 'keep-alive' })
  const id = \`chatcmpl-\${randomUUID()}\`
  const created = Math.floor(Date.now() / 1000)
  res.write(\`data: \${JSON.stringify({ id, object: 'chat.completion.chunk', created, model: modelId, choices: [{ index: 0, delta: { role: 'assistant', content: answer }, finish_reason: null }] })}\\n\\n\`)
  res.write(\`data: \${JSON.stringify({ id, object: 'chat.completion.chunk', created, model: modelId, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] })}\\n\\n\`)
  res.write('data: [DONE]\\n\\n')
  res.end()
}

/**
 * DSH's own stable per-conversation identity, when the model config for this
 * bridge turns on long prompt-cache retention (\`cacheRetention: 'long'\` plus
 * \`compat.supportsLongCacheRetention: true\` on the model entry in
 * \`settings.yaml\`) -- pi-ai then puts \`this.session.id\` (clamped to 64
 * chars) on every request as \`prompt_cache_key\`, a field this bridge doesn't
 * otherwise use for caching but repurposes here as a stable session key, so
 * a stateful n8n workflow (Data Table session lookup, agent memory) can
 * actually continue a conversation turn to turn instead of restarting it on
 * every message. Absent (config not set, or an older DSH build) rather than
 * an error -- the bridge still answers, just without continuity, same as
 * before this existed.
 */
function extractSessionId(body: Record<string, unknown>): string | undefined {
  return typeof body.prompt_cache_key === 'string' && body.prompt_cache_key !== '' ? body.prompt_cache_key : undefined
}

async function handleChatCompletions(ctx: Context, config: ResolvedConfig, req: IncomingMessage, res: ServerResponse): Promise<void> {
  if (req.method !== 'POST') { res.writeHead(405).end(); return }
  if (req.headers.authorization !== \`Bearer \${config.apiKey}\`) { res.writeHead(401).end(); return }
  const raw = await readRequestBody(req)
  const body = JSON.parse(raw) as Record<string, unknown>
  if (body.model !== config.modelId) {
    res.writeHead(400, { 'Content-Type': 'application/json' }).end(JSON.stringify({ error: { message: 'unknown model', type: 'model_not_found' } }))
    return
  }
  const question = extractLastUserText(body.messages)
  if (!question) { res.writeHead(400).end(); return }
  const sessionId = extractSessionId(body)

  // Session-title requests aren't real questions for the automation --
  // answer them locally (or via a real model, per \`titleMode\`, falling back
  // to local on any failure); never spend an n8n call on one.
  if (question.startsWith(SESSION_TITLE_REQUEST_PREFIX)) {
    const title = config.titleMode === 'llm'
      ? (await deriveSessionTitleViaLlm(ctx, config, question)) ?? deriveSessionTitle(question, config.sessionTitleMaxChars)
      : deriveSessionTitle(question, config.sessionTitleMaxChars)
    if (title === undefined) { res.writeHead(502).end(); return }
    writeSseAnswer(res, config.modelId, title)
    return
  }

  try {
    const answer = await fetchGroundedAnswer(config, question, sessionId)
    writeSseAnswer(res, config.modelId, answer)
  } catch (error) {
    res.writeHead(502, { 'Content-Type': 'application/json' }).end(JSON.stringify({ error: { message: String(error), type: 'upstream_error' } }))
  }
}

export class BridgeController extends Service {
  static inject = ['webServer']

  constructor(ctx: Context, rawConfig: BridgeConfig) {
    super(ctx, '${plan.packageName.replace(/-([a-z])/g, (_m, c: string) => c.toUpperCase())}Bridge')
    const config = resolveConfig(rawConfig)
    ctx.webServer.register({ kind: 'prefix', path: config.path, handler: (req, res) => handleChatCompletions(ctx, config, req, res) })
  }
}

export default BridgeController
`
}

// ---------------------------------------------------------------------------
// Phase 5 — install, generate, build
// ---------------------------------------------------------------------------

interface BuildOutcome {
  success: boolean
  log: string
}

/** Run one repo-root command, capturing combined output rather than throwing on a non-zero exit. */
async function runRepoCommand(
  repoRoot: string,
  command: string,
  args: string[],
): Promise<BuildOutcome> {
  try {
    const { stdout, stderr } = await execFileAsync(command, args, {
      cwd: repoRoot,
      maxBuffer: 16 * 1024 * 1024,
      shell: true,
    })
    return { success: true, log: `${stdout}${stderr}` }
  } catch (error) {
    const execError = error as {
      stdout?: string
      stderr?: string
      message: string
    }
    return {
      success: false,
      log: `${execError.stdout ?? ''}${execError.stderr ?? ''}${execError.message}`,
    }
  }
}

async function buildBridgePackage(
  repoRoot: string,
  packageName: string,
): Promise<BuildOutcome[]> {
  // Freshly-scaffolded packages under packages/custom/* are not wired into
  // tsconfig.host.json's own project-reference graph (same as this very
  // command's own package) — `tsc -b tsconfig.host.json` alone will never
  // compile them. A one-time scoped build populates their lib/types output
  // before the repo-wide build:lib:host bundles it with tsdown.
  const scopedTsconfig = path.join(
    'packages',
    'custom',
    packageName,
    'tsconfig.json',
  )
  const steps: [string, string[]][] = [
    ['pnpm', ['install']],
    ['pnpm', ['run', 'gen-tsconfig-paths']],
    [
      'node',
      [
        '--max-old-space-size=4096',
        './node_modules/typescript/bin/tsc',
        '-b',
        scopedTsconfig,
      ],
    ],
    ['pnpm', ['run', 'build:lib:host']],
  ]
  const outcomes: BuildOutcome[] = []
  for (const [command, args] of steps) {
    const outcome = await runRepoCommand(repoRoot, command, args)
    outcomes.push(outcome)
    if (!outcome.success) break
  }
  return outcomes
}

// ---------------------------------------------------------------------------
// Phase 6 — register the new model (cordis.patch.yml + settings.yaml)
// ---------------------------------------------------------------------------

/**
 * Back up a file to a sibling `<name>.backup-<date>` path before editing it,
 * mirroring the same manual safety net used for every other edit to these
 * runtime config files in this project.
 */
async function backupBeforeEdit(filePath: string): Promise<void> {
  const contents = await readFile(filePath, 'utf8')
  const stamp = new Date().toISOString().slice(0, 10)
  await writeFile(`${filePath}.backup-${stamp}`, contents, 'utf8')
}

/**
 * Locate the top-level `insert:` patch entry's item list in a parsed
 * cordis.patch.yml document, if the file has one yet.
 */
function findInsertItems(doc: ReturnType<typeof parseYamlDocument>): { items: unknown[] } | undefined {
  const topLevel = (doc.contents as { items?: unknown[] } | null)?.items ?? []
  const insertEntry = topLevel.find((item) => {
    const map = item as { get?: (key: string) => unknown }
    return typeof map.get === 'function' && map.get('insert') !== undefined
  }) as { get: (key: string) => { items: unknown[] } } | undefined
  return insertEntry?.get('insert')
}

/**
 * Read cordis.patch.yml (without writing anything) and report the webhook
 * URL already registered under the given package name, if any. Called
 * before any scaffolding or build work starts, so a genuine re-registration
 * (same workflow, same webhookUrl, refreshed) can be told apart from a name
 * collision (two different workflows whose titles happen to slugify to the
 * same package name) up front, instead of finding out at boot time the way
 * this used to.
 */
async function findExistingPatchWebhookUrl(
  profileDir: string,
  packageName: string,
): Promise<string | undefined> {
  const patchPath = path.join(profileDir, 'cordis.patch.yml')
  const doc = parseYamlDocument(await readFile(patchPath, 'utf8'))
  const items = findInsertItems(doc)?.items ?? []
  const existing = items.find((item) => {
    const map = item as { get?: (key: string) => unknown }
    return typeof map.get === 'function' && map.get('id') === packageName
  }) as { get: (key: string) => unknown } | undefined
  if (existing === undefined) return undefined

  const config = existing.get('config') as { get?: (key: string) => unknown } | undefined
  const webhookUrl = config?.get?.('webhookUrl')
  return typeof webhookUrl === 'string' ? webhookUrl : undefined
}

async function registerBridgeInPatch(
  profileDir: string,
  plan: BridgePlan,
): Promise<void> {
  const patchPath = path.join(profileDir, 'cordis.patch.yml')
  await backupBeforeEdit(patchPath)
  const doc = parseYamlDocument(await readFile(patchPath, 'utf8'))

  const newEntry = {
    id: plan.packageName,
    name: `@deepseek-ai/dsh-${plan.packageName}`,
    config: {
      webhookUrl: plan.webhookUrl,
      apiKey: plan.apiKey,
      modelId: plan.modelId,
    },
  }

  const insertList = findInsertItems(doc)
  if (insertList !== undefined) {
    const existingIndex = insertList.items.findIndex((item) => {
      const map = item as { get?: (key: string) => unknown }
      return typeof map.get === 'function' && map.get('id') === plan.packageName
    })
    // A matching id here always means "same workflow, re-registered" — a
    // genuine name collision (different webhookUrl) was already caught by
    // findExistingPatchWebhookUrl before any of this ran.
    if (existingIndex === -1) {
      insertList.items.push(doc.createNode(newEntry))
    } else {
      insertList.items[existingIndex] = doc.createNode(newEntry)
    }
  } else {
    doc.add(doc.createNode({ insert: [newEntry] }))
  }
  await writeFile(patchPath, String(doc), 'utf8')
}

async function registerBridgeInSettings(
  dshHomeDir: string,
  plan: BridgePlan,
  webServerPort: number,
): Promise<void> {
  const settingsPath = path.join(dshHomeDir, 'settings.yaml')
  await backupBeforeEdit(settingsPath)
  const doc = parseYamlDocument(await readFile(settingsPath, 'utf8'))
  doc.setIn(
    ['llm-pi-ai', 'providers', plan.packageName],
    doc.createNode({
      displayName: plan.packageName,
      apiKeyEnv: `${plan.packageName.toUpperCase().replace(/-/g, '_')}_API_KEY`,
      api: 'openai-completions',
      baseURL: `http://127.0.0.1:${webServerPort}/${plan.packageName}/v1/chat/completions`,
      models: [{ id: plan.modelId, name: plan.modelId }],
    }),
  )
  await writeFile(settingsPath, String(doc), 'utf8')
}

// ---------------------------------------------------------------------------
// Phase 7 — verify, tier one: the bridge's own logic against the live webhook
// ---------------------------------------------------------------------------

interface VerificationOutcome {
  success: boolean
  detail: string
}

/**
 * Exercise the exact fetch-and-parse logic just generated, directly against
 * the real n8n webhook — without booting DSH's own HTTP route, and without
 * touching the running DSH process at all (which this command cannot
 * restart while it is still executing inside it).
 */
async function verifyBridgeLogic(
  plan: BridgePlan,
): Promise<VerificationOutcome> {
  let response: Response
  try {
    response = await fetch(plan.webhookUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        chatInput:
          'This is a connection test from /model-creator — please reply with anything.',
      }),
    })
  } catch (error) {
    return {
      success: false,
      detail: `Could not reach the webhook: ${error instanceof Error ? error.message : String(error)}`,
    }
  }
  if (!response.ok) {
    return {
      success: false,
      detail: `Webhook returned HTTP ${response.status}.`,
    }
  }
  const raw = await response.text()
  let answer = raw
  if (plan.responseShape.customParseBody === undefined) {
    try {
      const data: unknown = JSON.parse(raw)
      if (typeof data === 'string') answer = data
      else if (
        data !== null &&
        typeof data === 'object' &&
        typeof (data as Record<string, unknown>).output === 'string'
      ) {
        answer = (data as Record<string, unknown>).output as string
      }
    } catch {
      // Not JSON — the raw body is the answer, matching the generated bridge's own fallback.
    }
  }
  // Deliberately evaluating the subagent's own generated parser the same way the generated package will run it.
  else
    answer = new Function('raw', plan.responseShape.customParseBody)(
      raw,
    ) as string
  answer = answer.trim()
  if (answer === '')
    return {
      success: false,
      detail: 'Webhook responded, but the parsed answer was empty.',
    }
  return {
    success: true,
    detail: `Webhook responded with a usable answer (${answer.length} characters).`,
  }
}

// ---------------------------------------------------------------------------
// Phase 8 — report back as a real reply
// ---------------------------------------------------------------------------

function buildReport(
  plan: BridgePlan,
  verification: VerificationOutcome,
): string {
  const lines = [
    `Built the "${plan.packageName}" model bridge from the workflow you gave me.`,
    '',
    verification.success
      ? `Verified: ${verification.detail}`
      : `Could not fully verify: ${verification.detail} — the package was still created; check the webhook before relying on it.`,
    '',
    'Two things only you can finish:',
    `1. Enter this key into Settings > Models for the "${plan.packageName}" provider: ${plan.apiKey}`,
    '2. Restart DSH, then pick the model from the dropdown and try it for real.',
  ]
  return lines.join('\n')
}

/**
 * Wrap the finished report in an explicit relay-verbatim instruction before
 * queuing it as a followup turn — the same pattern permaculture-command uses
 * for its own `buildRelayInstruction`. Without this wrapper, the model
 * receiving the bare report text treats "enter this key" / "restart DSH"
 * as an implicit request for itself to act on, then spends the reply
 * explaining that it has no tool to click Settings or restart the process —
 * the report was already correct and complete; the model's only job on
 * receiving it is to present it, not act on it.
 */
function buildRelayInstruction(report: string): string {
  return [
    '/model-creator has finished running. Its report to the user is below, already complete and correct.',
    'The two follow-up steps it lists (entering a key in Settings and restarting DSH) are for the person ' +
      'to do themselves — they are not a request for you to attempt those actions or to explain that you ' +
      'cannot. Reply with this report essentially verbatim — do not paraphrase, summarize, add commentary, ' +
      'or discuss your own tool limitations. Just present it as your reply:',
    '',
    report,
  ].join('\n')
}

function relayReport(
  agent: Agent,
  plan: BridgePlan,
  verification: VerificationOutcome,
): void {
  agent.followup(
    createUserMessage({
      content: [
        {
          type: 'text',
          text: buildRelayInstruction(buildReport(plan, verification)),
        },
      ],
      source: {
        kind: 'plugin',
        plugin: 'model-creator',
        form: 'notice',
        summary: boundContextSummary(
          `/model-creator finished building the "${plan.packageName}" model.`,
        ),
      },
    }),
  )
}

/** `ctx.modelCreator`: registers `/model-creator`. */
export class ModelCreatorController extends Service {
  private readonly config: ModelCreatorConfig

  constructor(ctx: Context, rawConfig: ModelCreatorConfig) {
    super(ctx, 'modelCreator')
    this.config = resolveConfig(rawConfig)

    ctx.inject(['commands', 'subagents', 'webServer'], (commandCtx) => {
      commandCtx.commands.register({
        name: 'model-creator',
        description: 'Build a new DSH custom model from an n8n webhook URL',
        input: { hint: '<n8n webhook URL>' },
        handler: async ({ agent, rawInput }) => {
          const trimmedInput = rawInput.trim()
          if (trimmedInput === '') {
            return { kind: 'error', text: 'Usage: /model-creator <n8n webhook URL>' }
          }
          if (/\s/.test(trimmedInput)) {
            return {
              kind: 'error',
              text: '/model-creator now takes just the webhook URL — no header-auth value needed anymore. Usage: /model-creator <n8n webhook URL>',
            }
          }
          const reference = trimmedInput

          const resolution = await resolveN8nWorkflow(this.config, reference)
          if (resolution.kind === 'error')
            return { kind: 'error', text: resolution.text }
          const { workflow } = resolution
          if (!workflow.active) {
            return {
              kind: 'error',
              text: `Workflow "${workflow.title}" is not published (Active) in n8n — publish it first.`,
            }
          }
          if (workflow.headerAuthAttached) {
            return {
              kind: 'error',
              text: `Workflow "${workflow.title}" has Header Auth configured on its trigger — /model-creator doesn't support per-request credentials. Open the workflow in n8n, set the Webhook node's Authentication to None, save, and run /model-creator again.`,
            }
          }

          const responseShape = await planResponseShape(
            commandCtx,
            agent,
            workflow,
          )
          const plan: BridgePlan = {
            packageName: deriveModelName(workflow.title),
            modelId: deriveModelName(workflow.title),
            webhookUrl: workflow.webhookUrl,
            apiKey: randomBytes(24).toString('base64url'),
            responseShape,
          }

          const existingWebhookUrl = await findExistingPatchWebhookUrl(
            this.config.profileDir,
            plan.packageName,
          )
          if (existingWebhookUrl !== undefined && existingWebhookUrl !== plan.webhookUrl) {
            return {
              kind: 'error',
              text: `A model named "${plan.packageName}" is already registered, pointing at a different webhook (${existingWebhookUrl}) than this one (${plan.webhookUrl}). If you meant to update that bridge, run /model-creator again with that same URL. If these are genuinely two different workflows, rename one of them in n8n — its title, not just the webhook path — so they don't produce the same model name, then try again.`,
            }
          }

          const packageDir = path.join(
            this.config.repoRoot,
            'packages',
            'custom',
            plan.packageName,
          )
          await mkdir(path.join(packageDir, 'src'), { recursive: true })
          await writeFile(
            path.join(packageDir, 'package.json'),
            renderBridgePackageJson(plan),
            'utf8',
          )
          await writeFile(
            path.join(packageDir, 'tsconfig.json'),
            renderBridgeTsconfig(),
            'utf8',
          )
          await writeFile(
            path.join(packageDir, 'src', 'index.ts'),
            renderBridgeSource(plan),
            'utf8',
          )

          const buildOutcomes = await buildBridgePackage(
            this.config.repoRoot,
            plan.packageName,
          )
          const failedStep = buildOutcomes.find(outcome => !outcome.success)
          if (failedStep !== undefined) {
            return {
              kind: 'error',
              text: `Build failed:\n${failedStep.log.slice(0, 2000)}`,
            }
          }

          await registerBridgeInPatch(this.config.profileDir, plan)
          await registerBridgeInSettings(
            this.config.dshHomeDir,
            plan,
            commandCtx.webServer.port,
          )

          const verification = await verifyBridgeLogic(plan)
          relayReport(agent, plan, verification)

          return {
            kind: 'success',
            text: `Built "${plan.packageName}" — see the reply below for what's left to finish.`,
          }
        },
      })
    })
  }
}

export default ModelCreatorController
