/**
 * A Cordis plugin that lets dsH's own model picker reach the n8n
 * "Permaculture Ethics & Principles Agent" webhook directly, as an ordinary
 * selectable model, with no slash command and no model call needed to
 * produce the answer.
 *
 * dsH's custom-provider transport (see `dsh-llm-pi-ai`) is streaming-only:
 * it speaks OpenAI Chat Completions over Server-Sent Events and has no
 * plain single-response HTTP mode. The n8n webhook this plugin calls
 * (`packages/custom/permaculture-command`'s target) answers with one
 * complete JSON body once its run finishes — n8n's own Webhook "Streaming"
 * response mode cannot substitute here: it hands the connection to n8n's
 * own execution-progress protocol (`begin`/`node-execute-*`/`item`/`end`
 * events), not to workflow-authored bytes, so a workflow cannot emit
 * arbitrary OpenAI-shaped SSE frames through it. This plugin is the piece
 * that actually speaks that wire format: it registers an
 * OpenAI-Chat-Completions-shaped route on the Web host's own HTTP server
 * (`ctx.webServer`, already running for the Web GUI), does one
 * deterministic, non-streaming fetch to the n8n webhook exactly like
 * `permaculture-command` already does, and wraps the finished, grounded
 * answer as a Server-Sent-Events chat-completion chunk stream. The n8n
 * webhook itself needs no change — both this plugin and the slash command
 * call it read-only, so they can coexist during the transition.
 *
 * This route answers its own detected session-title requests (see
 * `SESSION_TITLE_REQUEST_PREFIX` below) one of two ways, per `titleMode`:
 * `'local'` (the default) truncates the framed message text on a word
 * boundary, exactly as this route always did before `'llm'` existed;
 * `'llm'` instead asks a real model, over an OpenAI-Chat-Completions-shaped
 * endpoint such as OpenRouter, to summarize that text into a short title.
 * The `'llm'` path falls back to the same local truncation, for that one
 * request only, on any failure — bad or missing key, network error,
 * timeout, or a malformed response — so a title is always produced and a
 * bad deployment can never turn into a hard failure. Because `'local'` stays
 * the default and the fallback, rolling back a troublesome `'llm'` rollout
 * is a one-line config edit (`titleMode: local`), not a code change.
 *
 * A dsH custom provider then points its Base URL at this route
 * (`http://<webServer.host>:<webServer.port><config.path>`), so pi-ai's
 * `openai-completions` client reaches it exactly like any other gateway.
 *
 * Whichever model a session uses also gets asked, once per fresh session, to
 * summarize its first message into a session title — a fixed request shape
 * from `@deepseek-ai/dsh-session-title-llm` (see that package's
 * `frameMessages`), not a real permaculture question. Forwarding that
 * verbatim to n8n was confirmed (via the workflow's own execution log) to
 * both spend a live Google-Docs fetch and LLM call on a task with no
 * permaculture content, and to come back in the agent's own cited-answer
 * voice ("Session title: ... (Principle cited: ...)") rather than a short
 * plain title — a real, observed defect, not just a theoretical one. This
 * route recognizes that exact request shape and answers it locally from the
 * framed message text, with no n8n call at all.
 *
 * @module @deepseek-ai/dsh-permaculture-model-bridge
 */

import { randomUUID } from 'node:crypto'
import type { IncomingMessage, ServerResponse } from 'node:http'
import { Context, Service } from '@deepseek-ai/cordis'
import type {} from '@deepseek-ai/dsh-host-webserver'
// Also pulls in this module's `declare module '@deepseek-ai/cordis'`
// augmentation, which is what makes `ctx.get('credentials')` below typecheck.
import { credentialRef, isCredentialRefName } from '@deepseek-ai/dsh-credentials'

declare module '@deepseek-ai/cordis' {
  interface Context {
    permacultureModelBridge: PermacultureModelBridgeController
  }
}

/** Deployment-owned config: where to call, how to authenticate both hops, and which model id to answer for. */
export interface PermacultureModelBridgeConfig {
  /** The n8n Webhook Trigger URL for the Permaculture Ethics & Principles Agent workflow. */
  webhookUrl: string
  /** Extra headers sent with every call to the n8n webhook, e.g. `{ Authorization: 'Bearer <n8n-shared-secret>' }`. */
  webhookHeaders?: Record<string, string>
  /** Shared secret this route requires as `Authorization: Bearer <apiKey>` from the calling dsH custom provider. */
  apiKey: string
  /** The model id this route answers for; a request naming any other id is refused with `model_not_found`. */
  modelId: string
  /**
   * Absolute HTTP path this route is registered at on `ctx.webServer`, no trailing slash.
   * @default '/permaculture-model/v1/chat/completions'
   */
  path?: string
  /**
   * Character budget for the session title this route derives locally (see
   * `deriveSessionTitle`) instead of asking n8n. Longer first messages are
   * truncated to this length on a word boundary.
   * @default 60
   */
  sessionTitleMaxChars?: number
  /**
   * How this route derives the session title for its own detected
   * title-generation requests — see the module doc comment.
   * @default 'local'
   */
  titleMode?: 'local' | 'llm'
  /**
   * Name of the credential reference holding the API key sent as
   * `Authorization: Bearer <value>` to `titleLlmBaseUrl`. Required when
   * `titleMode` is `'llm'`; ignored otherwise. Holds a reference *name*, not
   * the key itself — matching how dsH's own provider configs
   * (`llm-pi-ai.providers.*.apiKeyEnv`) name secrets — so the key never
   * appears in this config or in loaded-config logging. Resolved through
   * `ctx.credentials` first, exactly like a built-in provider's own
   * `apiKeyEnv` (so a key already entered for, e.g., the `openrouter`
   * provider is found here too), then through a literal `process.env`
   * lookup if no credentials service is mounted.
   */
  titleLlmApiKeyEnv?: string
  /**
   * Model id sent to `titleLlmBaseUrl` for title generation. Only used
   * when `titleMode` is `'llm'`.
   * @default 'anthropic/claude-sonnet-5'
   */
  titleLlmModel?: string
  /**
   * OpenAI Chat Completions-compatible endpoint called for title
   * generation when `titleMode` is `'llm'`.
   * @default 'https://openrouter.ai/api/v1/chat/completions'
   */
  titleLlmBaseUrl?: string
  /**
   * Deadline for the `titleMode: 'llm'` request, in milliseconds. On
   * expiry the request is aborted and this route falls back to local
   * truncation for that one title, same as any other failure.
   * @default 15000
   */
  titleLlmTimeoutMs?: number
}

/** {@link PermacultureModelBridgeConfig} after validation, with defaults filled in (see individual fields' `@default`s above). */
interface ResolvedConfig {
  webhookUrl: string
  webhookHeaders?: Record<string, string>
  apiKey: string
  modelId: string
  path: string
  sessionTitleMaxChars: number
  titleMode: 'local' | 'llm'
  titleLlmApiKeyEnv?: string
  titleLlmModel: string
  titleLlmBaseUrl: string
  titleLlmTimeoutMs: number
}

const DEFAULT_PATH = '/permaculture-model/v1/chat/completions'
const DEFAULT_SESSION_TITLE_MAX_CHARS = 60
const DEFAULT_TITLE_MODE: 'local' | 'llm' = 'local'
const DEFAULT_TITLE_LLM_MODEL = 'anthropic/claude-sonnet-5'
const DEFAULT_TITLE_LLM_BASE_URL = 'https://openrouter.ai/api/v1/chat/completions'
const DEFAULT_TITLE_LLM_TIMEOUT_MS = 15000

/**
 * Validate deployment-owned config. Missing, blank, non-string, wrongly
 * shaped, or unknown fields fail at plugin load rather than being ignored.
 */
function resolveConfig(config: unknown): ResolvedConfig {
  // Validated as `unknown`, not `PermacultureModelBridgeConfig`: this config
  // is loaded from YAML and has never actually been checked against that
  // interface, so typing the parameter as the interface itself would let
  // TypeScript "prove" fields like `webhookHeaders` are already well-shaped
  // objects and flag the runtime checks below as unreachable dead code.
  if (typeof config !== 'object' || config === null) {
    throw new Error('PermacultureModelBridgeConfig must be an object')
  }
  const record = config as Record<string, unknown>
  const webhookUrl = record.webhookUrl
  if (typeof webhookUrl !== 'string' || webhookUrl.trim() === '') {
    throw new Error('PermacultureModelBridgeConfig needs a non-empty string `webhookUrl`')
  }
  const webhookHeaders = record.webhookHeaders
  if (webhookHeaders !== undefined && (typeof webhookHeaders !== 'object' || webhookHeaders === null || Array.isArray(webhookHeaders))) {
    throw new Error('PermacultureModelBridgeConfig `webhookHeaders`, if given, must be an object of string values')
  }
  const apiKey = record.apiKey
  if (typeof apiKey !== 'string' || apiKey.trim() === '') {
    throw new Error('PermacultureModelBridgeConfig needs a non-empty string `apiKey`')
  }
  const modelId = record.modelId
  if (typeof modelId !== 'string' || modelId.trim() === '') {
    throw new Error('PermacultureModelBridgeConfig needs a non-empty string `modelId`')
  }
  const path = record.path
  if (path !== undefined && (typeof path !== 'string' || path.trim() === '' || !path.startsWith('/') || path.endsWith('/'))) {
    throw new Error('PermacultureModelBridgeConfig `path`, if given, must be a non-empty string starting with "/" and without a trailing slash')
  }
  const sessionTitleMaxChars = record.sessionTitleMaxChars
  if (sessionTitleMaxChars !== undefined && (typeof sessionTitleMaxChars !== 'number' || !Number.isInteger(sessionTitleMaxChars) || sessionTitleMaxChars < 1)) {
    throw new Error('PermacultureModelBridgeConfig `sessionTitleMaxChars`, if given, must be a positive integer')
  }
  const titleMode = record.titleMode
  if (titleMode !== undefined && (typeof titleMode !== 'string' || (titleMode !== 'local' && titleMode !== 'llm'))) {
    throw new Error('PermacultureModelBridgeConfig `titleMode`, if given, must be "local" or "llm"')
  }
  const titleLlmApiKeyEnv = record.titleLlmApiKeyEnv
  if (titleLlmApiKeyEnv !== undefined && (typeof titleLlmApiKeyEnv !== 'string' || titleLlmApiKeyEnv.trim() === '')) {
    throw new Error('PermacultureModelBridgeConfig `titleLlmApiKeyEnv`, if given, must be a non-empty string')
  }
  if (titleMode === 'llm' && titleLlmApiKeyEnv === undefined) {
    throw new Error('PermacultureModelBridgeConfig needs `titleLlmApiKeyEnv` (the name of an env var holding the key) when `titleMode` is "llm"')
  }
  const titleLlmModel = record.titleLlmModel
  if (titleLlmModel !== undefined && (typeof titleLlmModel !== 'string' || titleLlmModel.trim() === '')) {
    throw new Error('PermacultureModelBridgeConfig `titleLlmModel`, if given, must be a non-empty string')
  }
  const titleLlmBaseUrl = record.titleLlmBaseUrl
  if (titleLlmBaseUrl !== undefined && (typeof titleLlmBaseUrl !== 'string' || titleLlmBaseUrl.trim() === '')) {
    throw new Error('PermacultureModelBridgeConfig `titleLlmBaseUrl`, if given, must be a non-empty string')
  }
  const titleLlmTimeoutMs = record.titleLlmTimeoutMs
  if (titleLlmTimeoutMs !== undefined && (typeof titleLlmTimeoutMs !== 'number' || !Number.isInteger(titleLlmTimeoutMs) || titleLlmTimeoutMs < 1)) {
    throw new Error('PermacultureModelBridgeConfig `titleLlmTimeoutMs`, if given, must be a positive integer')
  }
  const unknownKeys = Object.keys(record).filter(key =>
    key !== 'webhookUrl' && key !== 'webhookHeaders' && key !== 'apiKey' && key !== 'modelId' && key !== 'path' && key !== 'sessionTitleMaxChars'
    && key !== 'titleMode' && key !== 'titleLlmApiKeyEnv' && key !== 'titleLlmModel' && key !== 'titleLlmBaseUrl' && key !== 'titleLlmTimeoutMs')
  if (unknownKeys.length > 0) {
    throw new Error(`PermacultureModelBridgeConfig has unknown key(s) ${unknownKeys.join(', ')} — config is { webhookUrl, webhookHeaders?, apiKey, modelId, path?, sessionTitleMaxChars?, titleMode?, titleLlmApiKeyEnv?, titleLlmModel?, titleLlmBaseUrl?, titleLlmTimeoutMs? }`)
  }
  // exactOptionalPropertyTypes: only include `webhookHeaders`/`titleLlmApiKeyEnv`
  // when actually given — assigning the key `undefined` is a type error here.
  return {
    webhookUrl,
    apiKey,
    modelId,
    path: path ?? DEFAULT_PATH,
    sessionTitleMaxChars: sessionTitleMaxChars ?? DEFAULT_SESSION_TITLE_MAX_CHARS,
    titleMode: (titleMode as 'local' | 'llm' | undefined) ?? DEFAULT_TITLE_MODE,
    titleLlmModel: titleLlmModel ?? DEFAULT_TITLE_LLM_MODEL,
    titleLlmBaseUrl: titleLlmBaseUrl ?? DEFAULT_TITLE_LLM_BASE_URL,
    titleLlmTimeoutMs: titleLlmTimeoutMs ?? DEFAULT_TITLE_LLM_TIMEOUT_MS,
    ...webhookHeaders === undefined ? {} : { webhookHeaders: webhookHeaders as Record<string, string> },
    ...titleLlmApiKeyEnv === undefined ? {} : { titleLlmApiKeyEnv },
  }
}

/** Outcome of the n8n webhook call: either the trimmed grounded answer, or an error message to report to the caller. */
type FetchOutcome =
  | { kind: 'answer'; answer: string }
  | { kind: 'error'; text: string }

/**
 * Call the n8n webhook and interpret its response. Identical in shape to
 * `permaculture-command`'s `fetchGroundedAnswer` — both call the same
 * webhook and parse the same `{ output: "<answer text>" }` shape — kept
 * as a separate copy here so this plugin stays self-contained and
 * independently removable, since the slash command it parallels is
 * intended to be retired once this route is confirmed working.
 */
async function fetchGroundedAnswer(config: ResolvedConfig, question: string): Promise<FetchOutcome> {
  let response: Response
  try {
    response = await fetch(config.webhookUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...config.webhookHeaders },
      body: JSON.stringify({ chatInput: question }),
    })
  } catch (error) {
    return {
      kind: 'error',
      text: `Could not reach the n8n webhook: ${error instanceof Error ? error.message : String(error)}`,
    }
  }

  const raw = await response.text().catch(() => '')

  if (!response.ok) {
    return {
      kind: 'error',
      text: `n8n webhook returned HTTP ${response.status}${raw ? `: ${raw.slice(0, 500)}` : ''}`,
    }
  }

  let answer = raw
  try {
    const data: unknown = JSON.parse(raw)
    if (typeof data === 'string') {
      answer = data
    } else if (data !== null && typeof data === 'object' && typeof (data as Record<string, unknown>).output === 'string') {
      answer = (data as Record<string, unknown>).output as string
    }
  } catch {
    // Not JSON — fall back to the raw response body as-is.
  }
  answer = answer.trim()

  if (answer === '') {
    return { kind: 'error', text: 'The n8n webhook returned an empty response.' }
  }

  return { kind: 'answer', answer }
}

/** Read a request body to completion as a UTF-8 string. */
function readRequestBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = []
    req.on('data', (chunk: Buffer) => { chunks.push(chunk) })
    req.on('end', () => { resolve(Buffer.concat(chunks).toString('utf8')) })
    req.on('error', (error: Error) => { reject(error) })
  })
}

/** Join the `text` parts of an OpenAI-style message `content` field, which may be a plain string or a content-part array. */
function extractTextContent(content: unknown): string | undefined {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return undefined
  const parts: string[] = []
  for (const part of content) {
    if (part === null || typeof part !== 'object') continue
    const record = part as Record<string, unknown>
    if (record.type === 'text' && typeof record.text === 'string') parts.push(record.text)
  }
  return parts.length === 0 ? undefined : parts.join('\n')
}

/**
 * Prefixes DSH's own agent loop appends, as additional trailing `role:
 * "user"` messages, after the person's real turn — a "workspace instructions"
 * reminder and a "current runtime context" snapshot. Confirmed by inspecting
 * the actual n8n webhook payloads this bridge sent: a naive "just take the
 * last user message" extraction was picking up the runtime-context snapshot
 * verbatim instead of what the person actually typed, so every real chat
 * turn was answering the wrong question. DSH may add more such synthetic
 * messages over time; widen this list rather than assume the last `user`
 * message is always the real one.
 */
const SYNTHETIC_CONTEXT_PREFIXES = ['<system-reminder>', 'Current runtime context.']

/** Whether an extracted user-turn text is one of DSH's own injected context messages rather than something the person typed. */
function isSyntheticContext(text: string): boolean {
  return SYNTHETIC_CONTEXT_PREFIXES.some(prefix => text.startsWith(prefix))
}

/**
 * Find the last `role: "user"` message that looks like the person's own
 * words, skipping DSH's own injected context messages (see
 * {@link SYNTHETIC_CONTEXT_PREFIXES}), and extract its text.
 */
function extractLastUserText(messages: unknown): string | undefined {
  if (!Array.isArray(messages)) return undefined
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message: unknown = messages[index]
    if (message === null || typeof message !== 'object') continue
    const record = message as Record<string, unknown>
    if (record.role !== 'user') continue
    const text = extractTextContent(record.content)
    if (text === undefined || isSyntheticContext(text)) continue
    return text
  }
  return undefined
}

/**
 * The exact, stable prefix `@deepseek-ai/dsh-session-title-llm` frames its
 * one auxiliary request with (its `frameMessages`: a fixed instruction
 * followed by `JSON.stringify` of the session's selected human messages).
 * Detecting this lets `handleChatCompletions` answer session-title requests
 * locally instead of forwarding them to n8n.
 */
const SESSION_TITLE_REQUEST_PREFIX = 'Generate the session title from this JSON array of human messages:\n'

/** One entry of the JSON array `dsh-session-title-llm` frames its request with: `{ seq, text }` per selected human message. */
interface SessionTitleSourceMessage {
  readonly text?: unknown
}

/**
 * Parse a detected session-title request and return its first selected
 * message's text, whitespace-collapsed and trimmed — mirroring the
 * first-prompt title provider's own choice of message. Shared by the local
 * truncation path and the `titleMode: 'llm'` path below, so both derive a
 * title from exactly the same source text. Returns `undefined` when the
 * request's JSON array is missing, malformed, or has no usable text.
 */
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
  const collapsed = first.text.replace(/\s+/g, ' ').trim()
  return collapsed === '' ? undefined : collapsed
}

/** Truncate `text` to `maxChars` on a word boundary (rather than mid-word), appending an ellipsis when cut. Leaves short text untouched. */
function truncateTitle(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text
  const truncated = text.slice(0, maxChars)
  const lastSpace = truncated.lastIndexOf(' ')
  return `${lastSpace > 0 ? truncated.slice(0, lastSpace) : truncated}…`
}

/**
 * Derive a short, plain-text session title locally from a detected
 * session-title request, by truncating its first selected message —
 * without this bridge needing to know the configured word-count target.
 * This is both the `titleMode: 'local'` behavior and the fallback
 * `titleMode: 'llm'` uses when the real-LLM call fails for any reason.
 * Returns `undefined` when {@link extractSessionTitleSourceText} can't find
 * usable text, so the caller can fail the request cleanly rather than guess
 * at a title.
 */
function deriveSessionTitle(requestText: string, maxChars: number): string | undefined {
  const collapsed = extractSessionTitleSourceText(requestText)
  return collapsed === undefined ? undefined : truncateTitle(collapsed, maxChars)
}

/**
 * The system prompt sent with every `titleMode: 'llm'` title request. Kept
 * short and single-purpose so the call is cheap and its output needs
 * minimal cleanup.
 */
const TITLE_LLM_SYSTEM_PROMPT = 'Write a short session title (a few words, plain text, no quotes, no trailing punctuation) that summarizes the user message that follows. Reply with only the title — no preamble, no explanation.'

/**
 * Resolve `name` to a secret value the same way dsH's own providers do (see
 * `llm-pi-ai`'s `authContextFrom`): through `ctx.credentials` first — the
 * seam backing `.dsh/.credentials.yaml`'s `refs`, which is where dsH
 * actually stores a key entered for a provider like `apiKeyEnv:
 * OPENROUTER_API_KEY`, not the OS process environment — falling back to a
 * literal `process.env[name]` when no credentials service is mounted (a
 * bare env var still works then) or it has nothing stored under that name.
 * Never throws: a malformed name, a service with no such service mounted,
 * or a resolution error is all just "not found" here, matching this
 * route's blanket policy of always falling back rather than failing.
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
 * Ask a real model — `config.titleLlmModel` over `config.titleLlmBaseUrl`,
 * an OpenAI Chat-Completions-compatible endpoint such as OpenRouter — to
 * summarize a detected session-title request's first message into a short
 * title. Returns `undefined` on ANY failure: the configured key isn't
 * resolvable, a network error, a non-OK HTTP status, a malformed or empty
 * response body, or the request timing out at `config.titleLlmTimeoutMs` —
 * so the caller can always fall back to {@link deriveSessionTitle}'s local
 * truncation and still produce a title. Never throws.
 */
async function deriveSessionTitleViaLlm(ctx: Context, config: ResolvedConfig, requestText: string): Promise<string | undefined> {
  if (config.titleLlmApiKeyEnv === undefined) return undefined
  const apiKey = await resolveTitleLlmApiKey(ctx, config.titleLlmApiKeyEnv)
  if (!apiKey) return undefined

  const sourceText = extractSessionTitleSourceText(requestText)
  if (sourceText === undefined) return undefined

  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), config.titleLlmTimeoutMs)
  try {
    const response = await fetch(config.titleLlmBaseUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
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

    // Strip whitespace and a wrapping quote pair some models add despite
    // the system prompt asking for neither.
    const collapsed = content.replace(/\s+/g, ' ').trim().replace(/^["']|["']$/g, '')
    if (collapsed === '') return undefined
    // Still cap length locally: a real model can ignore the word-count ask.
    return truncateTitle(collapsed, config.sessionTitleMaxChars)
  } catch {
    // Network error, abort (timeout), or anything else unexpected — treat
    // it the same as a bad response and let the caller fall back.
    return undefined
  } finally {
    clearTimeout(timer)
  }
}

/** Write one OpenAI-style JSON error response. Only valid before any bytes of the SSE response have been sent. */
function sendJsonError(res: ServerResponse, status: number, type: string, message: string): void {
  const body = JSON.stringify({ error: { message, type } })
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(body),
  })
  res.end(body)
}

/**
 * Write the finished answer as an OpenAI Chat Completions SSE stream: one
 * content-delta chunk, one closing `finish_reason: "stop"` chunk, then
 * `data: [DONE]`. The whole answer is already in hand from one completed
 * n8n run, so this sends it as a single delta rather than token-by-token —
 * satisfying the transport's SSE requirement without depending on n8n's
 * own (non-conforming) streaming support.
 */
function writeSseAnswer(res: ServerResponse, modelId: string, answer: string): void {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  })

  const id = `chatcmpl-${randomUUID()}`
  const created = Math.floor(Date.now() / 1000)
  const contentChunk = {
    id,
    object: 'chat.completion.chunk',
    created,
    model: modelId,
    choices: [{ index: 0, delta: { role: 'assistant', content: answer }, finish_reason: null }],
  }
  const finishChunk = {
    id,
    object: 'chat.completion.chunk',
    created,
    model: modelId,
    choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
  }

  res.write(`data: ${JSON.stringify(contentChunk)}\n\n`)
  res.write(`data: ${JSON.stringify(finishChunk)}\n\n`)
  res.write('data: [DONE]\n\n')
  res.end()
}

/**
 * Handle one OpenAI Chat Completions request: authenticate, extract the
 * last user message, fetch the grounded answer from n8n, and answer with
 * either an SSE chunk stream or a JSON error — never both for the same
 * request, so a failure before the answer is in hand reports as an
 * ordinary HTTP error instead of failing mid-stream.
 */
async function handleChatCompletions(ctx: Context, config: ResolvedConfig, req: IncomingMessage, res: ServerResponse): Promise<void> {
  if (req.method !== 'POST') {
    sendJsonError(res, 405, 'method_not_allowed', 'Only POST is supported.')
    return
  }

  if (req.headers.authorization !== `Bearer ${config.apiKey}`) {
    sendJsonError(res, 401, 'invalid_api_key', 'Missing or incorrect bearer token.')
    return
  }

  let raw: string
  try {
    raw = await readRequestBody(req)
  } catch (error) {
    sendJsonError(res, 400, 'invalid_request_error', `Could not read the request body: ${error instanceof Error ? error.message : String(error)}`)
    return
  }

  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    sendJsonError(res, 400, 'invalid_request_error', 'Request body is not valid JSON.')
    return
  }
  if (parsed === null || typeof parsed !== 'object') {
    sendJsonError(res, 400, 'invalid_request_error', 'Request body must be a JSON object.')
    return
  }
  const body = parsed as Record<string, unknown>

  if (body.model !== config.modelId) {
    sendJsonError(res, 400, 'model_not_found', `This endpoint only serves model "${config.modelId}".`)
    return
  }

  const question = extractLastUserText(body.messages)?.trim()
  if (question === undefined || question === '') {
    sendJsonError(res, 400, 'invalid_request_error', 'No non-empty `role: "user"` message found in `messages`.')
    return
  }

  // Session-title requests aren't permaculture questions — see the module
  // doc comment. Answer them locally (or via a real LLM, per `titleMode`,
  // falling back to local on any failure); never spend an n8n call on one.
  if (question.startsWith(SESSION_TITLE_REQUEST_PREFIX)) {
    const title = config.titleMode === 'llm'
      ? (await deriveSessionTitleViaLlm(ctx, config, question)) ?? deriveSessionTitle(question, config.sessionTitleMaxChars)
      : deriveSessionTitle(question, config.sessionTitleMaxChars)
    if (title === undefined) {
      sendJsonError(res, 502, 'upstream_error', 'Could not derive a session title locally from the request.')
      return
    }
    writeSseAnswer(res, config.modelId, title)
    return
  }

  const outcome = await fetchGroundedAnswer(config, question)
  if (outcome.kind === 'error') {
    sendJsonError(res, 502, 'upstream_error', outcome.text)
    return
  }

  writeSseAnswer(res, config.modelId, outcome.answer)
}

/**
 * `ctx.permacultureModelBridge`: registers the OpenAI-Chat-Completions
 * route on `ctx.webServer` that a dsH custom provider's Base URL points
 * at.
 */
export class PermacultureModelBridgeController extends Service {
  static inject = ['webServer']

  private readonly config: ResolvedConfig

  constructor(ctx: Context, rawConfig: PermacultureModelBridgeConfig) {
    super(ctx, 'permacultureModelBridge')
    this.config = resolveConfig(rawConfig)
    const config = this.config

    // 'prefix', not 'exact': dsH's setup guide documents the custom-provider
    // Base URL as the webhook's exact complete address with nothing appended,
    // but pi-ai's OpenAI-compatible client (like most OpenAI SDKs) may instead
    // treat baseURL as an API root and append "/chat/completions" itself.
    // Registering this route as a prefix match answers correctly either way,
    // without needing to pin down which convention pi-ai actually follows.
    ctx.webServer.register({
      kind: 'prefix',
      path: config.path,
      handler: (req, res) => handleChatCompletions(ctx, config, req, res),
    })
  }
}

export default PermacultureModelBridgeController
