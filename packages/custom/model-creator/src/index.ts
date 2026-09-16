/**
 * A Cordis plugin that registers the `/model-creator` slash command: given an
 * n8n workflow link, it builds a brand-new DSH custom-model bridge package
 * for that workflow end to end — resolve the workflow's webhook via n8n's
 * own REST API, scaffold a `packages/custom/<derived-name>/` package from
 * the standard bridge template, install and build it, register it in the
 * profile's `cordis.patch.yml` and `settings.yaml`, verify the generated
 * bridge logic directly against the live webhook, and report back what is
 * left for a person to finish by hand.
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
// Phase 2 — resolve the n8n workflow from a link or bare id (deterministic)
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

/** Pull a bare workflow id out of a pasted n8n URL, or accept a bare id as-is. */
function extractWorkflowId(reference: string): string | undefined {
  const trimmed = reference.trim()
  const urlMatch = /\/workflow\/([A-Za-z0-9_-]+)/.exec(trimmed)
  if (urlMatch?.[1] !== undefined) return urlMatch[1]
  if (/^[A-Za-z0-9_-]+$/.test(trimmed)) return trimmed
  return undefined
}

/**
 * Fetch a workflow from n8n's REST API and pull out what's needed to build a
 * bridge for it: its title, its Webhook Trigger node's Production URL and
 * published state, and whether Header Auth is attached (existence only — the
 * actual secret value is never exposed by this API and must come from the
 * person running the command).
 */
async function resolveN8nWorkflow(
  config: ModelCreatorConfig,
  reference: string,
): Promise<WorkflowResolution> {
  const workflowId = extractWorkflowId(reference)
  if (workflowId === undefined) {
    return {
      kind: 'error',
      text: `Could not find a workflow id in "${reference}" — pass the workflow's URL or its bare id.`,
    }
  }

  let response: Response
  try {
    response = await fetch(
      `${config.n8nBaseUrl}/api/v1/workflows/${workflowId}`,
      {
        headers: { 'X-N8N-API-KEY': config.n8nApiKey },
      },
    )
  } catch (error) {
    return {
      kind: 'error',
      text: `Could not reach n8n at ${config.n8nBaseUrl}: ${error instanceof Error ? error.message : String(error)}`,
    }
  }
  if (!response.ok) {
    return {
      kind: 'error',
      text: `n8n returned HTTP ${response.status} for workflow "${workflowId}" — check the id and that the API key can read it.`,
    }
  }

  const data = (await response.json()) as Record<string, unknown>
  const nodes = Array.isArray(data.nodes)
    ? (data.nodes as Record<string, unknown>[])
    : []
  const triggerNode = nodes.find(
    node => typeof node.type === 'string' && node.type.includes('webhook'),
  )
  if (triggerNode === undefined) {
    return {
      kind: 'error',
      text: 'This workflow has no Webhook Trigger node — model-creator only builds bridges for webhook-triggered workflows.',
    }
  }

  const triggerParams = (triggerNode.parameters ?? {}) as Record<
    string,
    unknown
  >
  const webhookPath =
    typeof triggerParams.path === 'string' ? triggerParams.path : workflowId
  const webhookUrl = `${config.n8nBaseUrl}/webhook/${webhookPath}`
  const headerAuthAttached = triggerParams.authentication === 'headerAuth'
  const title = typeof data.name === 'string' ? data.name : workflowId
  const active = data.active === true

  // Only the trigger and the terminal (last) node go to the subagent in
  // Phase 3 — not the whole workflow, which can contain unrelated internal
  // logic the response-shape question has no need to see.
  const terminalNode = nodes.length > 0 ? nodes[nodes.length - 1] : undefined
  const relevantNodes = { trigger: triggerNode, terminal: terminalNode }

  return {
    kind: 'resolved',
    workflow: {
      workflowId,
      title,
      webhookUrl,
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
  headerAuthValue: string | undefined
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

declare module '@deepseek-ai/cordis' {
  interface Context {
    ${plan.packageName.replace(/-([a-z])/g, (_m, c: string) => c.toUpperCase())}Bridge: BridgeController
  }
}

export interface BridgeConfig {
  webhookUrl: string
  webhookHeaders?: Record<string, string>
  apiKey: string
  modelId: string
  path?: string
}

interface ResolvedConfig {
  webhookUrl: string
  webhookHeaders?: Record<string, string>
  apiKey: string
  modelId: string
  path: string
}

const DEFAULT_PATH = '/${plan.packageName}/v1/chat/completions'

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
  return {
    webhookUrl,
    apiKey,
    modelId,
    path: (path as string | undefined) ?? DEFAULT_PATH,
    ...record.webhookHeaders === undefined ? {} : { webhookHeaders: record.webhookHeaders as Record<string, string> },
  }
}

/** Extracts the answer text from the webhook's raw response body. */
function parseAnswer(raw: string): string {
${parseBody
  .split('\n')
  .map(line => `  ${line}`)
  .join('\n')}
}

async function fetchGroundedAnswer(config: ResolvedConfig, question: string): Promise<string> {
  const response = await fetch(config.webhookUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...config.webhookHeaders },
    body: JSON.stringify({ chatInput: question }),
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

/** Join the \`text\` parts of an OpenAI-style message \`content\` field, which may be a plain string or a content-part array. */
function extractTextContent(content: unknown): string | undefined {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return undefined
  const parts: string[] = []
  for (const part of content) {
    if (part === null || typeof part !== 'object') continue
    const record = part as Record<string, unknown>
    if (record.type === 'text' && typeof record.text === 'string') parts.push(record.text)
  }
  return parts.length === 0 ? undefined : parts.join('\\n')
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

function writeSseAnswer(res: ServerResponse, modelId: string, answer: string): void {
  res.writeHead(200, { 'Content-Type': 'text/event-stream; charset=utf-8', 'Cache-Control': 'no-cache', Connection: 'keep-alive' })
  const id = \`chatcmpl-\${randomUUID()}\`
  const created = Math.floor(Date.now() / 1000)
  res.write(\`data: \${JSON.stringify({ id, object: 'chat.completion.chunk', created, model: modelId, choices: [{ index: 0, delta: { role: 'assistant', content: answer }, finish_reason: null }] })}\\n\\n\`)
  res.write(\`data: \${JSON.stringify({ id, object: 'chat.completion.chunk', created, model: modelId, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] })}\\n\\n\`)
  res.write('data: [DONE]\\n\\n')
  res.end()
}

async function handleChatCompletions(config: ResolvedConfig, req: IncomingMessage, res: ServerResponse): Promise<void> {
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
  try {
    const answer = await fetchGroundedAnswer(config, question)
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
    ctx.webServer.register({ kind: 'prefix', path: config.path, handler: (req, res) => handleChatCompletions(config, req, res) })
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

async function registerBridgeInPatch(
  profileDir: string,
  plan: BridgePlan,
): Promise<void> {
  const patchPath = path.join(profileDir, 'cordis.patch.yml')
  await backupBeforeEdit(patchPath)
  const doc = parseYamlDocument(await readFile(patchPath, 'utf8'))
  const contents = doc.contents
  const topLevel = (contents as { items?: unknown[] } | null)?.items ?? []
  const insertEntry = topLevel.find((item) => {
    const map = item as { get?: (key: string) => unknown }
    return typeof map.get === 'function' && map.get('insert') !== undefined
  }) as { get: (key: string) => { items: unknown[] } } | undefined

  const newEntry = {
    id: plan.packageName,
    name: `@deepseek-ai/dsh-${plan.packageName}`,
    config: {
      webhookUrl: plan.webhookUrl,
      apiKey: plan.apiKey,
      modelId: plan.modelId,
      ...(plan.headerAuthValue === undefined
        ? {}
        : {
          webhookHeaders: { Authorization: `Bearer ${plan.headerAuthValue}` },
        }),
    },
  }

  if (insertEntry !== undefined) {
    insertEntry.get('insert').items.push(doc.createNode(newEntry))
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
      headers: {
        'Content-Type': 'application/json',
        ...(plan.headerAuthValue === undefined
          ? {}
          : { Authorization: `Bearer ${plan.headerAuthValue}` }),
      },
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
        description: 'Build a new DSH custom model from an n8n workflow link',
        input: { hint: '<n8n workflow URL or id> [header-auth-value]' },
        handler: async ({ agent, rawInput }) => {
          const [reference, headerAuthValue] = rawInput.trim().split(/\s+/, 2)
          if (reference === undefined || reference === '') {
            return {
              kind: 'error',
              text: 'Usage: /model-creator <n8n workflow URL or id> [header-auth-value]',
            }
          }

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
          if (workflow.headerAuthAttached && headerAuthValue === undefined) {
            return {
              kind: 'error',
              text: `Workflow "${workflow.title}" has Header Auth configured — pass its value: /model-creator ${reference} <value>`,
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
            headerAuthValue,
            apiKey: randomBytes(24).toString('base64url'),
            responseShape,
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
