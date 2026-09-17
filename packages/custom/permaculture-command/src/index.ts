/**
 * A Cordis plugin that registers the `/permaculture-help` slash command and,
 * while its thread is open, a model-callable follow-up tool. Both paths call
 * the n8n "Permaculture Ethics & Principles Agent" workflow's Webhook
 * Trigger directly over HTTP — the trigger, the lookup, and the fetch stay
 * fully deterministic, with no model call involved in producing the answer.
 *
 * What happens with that answer is the one deliberate exception: instead of
 * only returning it as the command's (or tool's) own result text (which
 * DSH's Web UI renders as a small collapsed row, never as a normal chat
 * message — see packages/interaction/commands/README.md and
 * packages/client/ui-commands/README.md), both paths also call
 * `agent.followup(...)` with the already-fetched answer and a strict
 * relay-verbatim instruction, sourced as a collapsed `notice` — the same
 * shape `webhook` and `schedule` use for external content — so only a short
 * summary, never the full instruction text, ever appears in the
 * conversation's own message bubble. `followup` queues that message as the
 * sole content of a fresh turn, producing one real, normal-looking chat
 * reply.
 *
 * `/permaculture-help <question>` also opens a durable per-session "thread":
 * while it is open, a system-prompt section tells the model that on-topic
 * follow-ups — including answering the reference agent's own clarifying
 * question — should go through the `ask_permaculture_followup` tool rather
 * than declining or asking the user to retype the command. That tool reuses
 * the exact same fetch-and-relay path, so the no-paraphrase guarantee holds
 * for follow-ups too, and calls `exec.concludeTurn()` so the model cannot
 * also restate the answer itself in the same turn — the relay arrives as its
 * own fresh turn instead. `/permaculture-help off` closes the thread.
 *
 * Modeled on `/plan` (packages/plan/plan-mode/src/index.ts), but without its
 * `WeakMap` + `agent/pre-step` deferred-commit machinery: that exists there
 * to keep plan mode consistent for an already-open turn the user is actively
 * steering. This thread only gates a prompt section and a tool's
 * availability, so toggling it mid-turn and having it take effect one
 * request later is harmless — the command and the tool append directly.
 *
 * @module @deepseek-ai/dsh-permaculture-command
 */

import { Context, Service } from '@deepseek-ai/cordis'
import { z as zod } from 'zod'
import type { ZodType } from 'zod'
import type { Agent } from '@deepseek-ai/dsh-agent'
import { boundContextSummary, createUserMessage } from '@deepseek-ai/dsh-llm'
import type { Session } from '@deepseek-ai/dsh-session'
import { defineTool } from '@deepseek-ai/dsh-tools'
import type {} from '@deepseek-ai/dsh-system-prompt'
import type {} from '@deepseek-ai/dsh-session-projection'
import type { ProjectionDefinition } from '@deepseek-ai/dsh-session-projection'
import type {} from '@deepseek-ai/dsh-commands'

declare module '@deepseek-ai/dsh-session/types' {
  interface SessionEventMap {
    /**
     * Whether a Permaculture Ethics & Principles follow-up thread is open
     * from this point on: log-only, non-surface, whole-value replace. The
     * last `permaculture/thread` wins; a log with none folds to inactive
     * through the projection unit's fold.
     *
     * Declared outside the repo (`packages/custom/`), so it is never in the
     * generated `KNOWN_SESSION_EVENT_TYPES` vocabulary a first-party reader
     * checks a persisted log against. Both append sites below pass
     * `{ ignorable: true }` so a build that does not know this type can skip
     * it on reload instead of refusing the whole session — losing it just
     * folds back to the default `active: false`, which is safe here since
     * this event only gates a prompt section and a tool's availability.
     */
    'permaculture/thread': { active: boolean }
  }
}

declare module '@deepseek-ai/dsh-session-projection/types' {
  interface SessionProjectionStateMap {
    /** Host-only fold of the logged `permaculture/thread` events. */
    permacultureThread: PermacultureThreadState
  }
}

declare module '@deepseek-ai/cordis' {
  interface Context {
    permacultureCommand: PermacultureCommandController
  }
}

/** Session-scoped state folded from the durable log. */
export interface PermacultureThreadState {
  active: boolean
}

/**
 * The model-facing follow-up tool's name. It stays registered while the
 * thread is inactive so the request tool catalog is stable across
 * transitions, matching `exit_plan_mode`'s own design.
 */
export const ASK_PERMACULTURE_FOLLOWUP = 'ask_permaculture_followup'

/** Deployment-owned config: where to call, and how to authenticate. */
export interface PermacultureCommandConfig {
  /** The n8n Webhook Trigger URL for the Permaculture Ethics & Principles Agent workflow. */
  webhookUrl: string
  /** Extra headers sent with every call, e.g. `{ Authorization: 'Bearer <token>' }`. */
  headers?: Record<string, string>
}

/**
 * Validate deployment-owned config. Missing, blank, non-string, or unknown
 * fields fail at plugin load rather than being ignored.
 */
function resolveConfig(config: unknown): PermacultureCommandConfig {
  // Validated as `unknown`, not `PermacultureCommandConfig`: this config is
  // loaded from YAML and has never actually been checked against that
  // interface, so typing the parameter as the interface itself would let
  // TypeScript "prove" `headers` is already a well-shaped object and flag
  // the runtime check below as unreachable dead code.
  if (typeof config !== 'object' || config === null) {
    throw new Error('PermacultureCommandConfig must be an object')
  }
  const record = config as Record<string, unknown>
  const webhookUrl = record.webhookUrl
  if (typeof webhookUrl !== 'string' || webhookUrl.trim() === '') {
    throw new Error('PermacultureCommandConfig needs a non-empty string `webhookUrl`')
  }
  const headers = record.headers
  if (headers !== undefined && (typeof headers !== 'object' || headers === null || Array.isArray(headers))) {
    throw new Error('PermacultureCommandConfig `headers`, if given, must be an object of string values')
  }
  const unknownKeys = Object.keys(record).filter(key => key !== 'webhookUrl' && key !== 'headers')
  if (unknownKeys.length > 0) {
    throw new Error(`PermacultureCommandConfig has unknown key(s) ${unknownKeys.join(', ')} — config is { webhookUrl, headers? }`)
  }
  // exactOptionalPropertyTypes: only include `headers` at all when it was
  // actually given — assigning the key `undefined` is a type error here.
  return headers === undefined ? { webhookUrl } : { webhookUrl, headers: headers as Record<string, string> }
}

/**
 * Builds the relayed instruction: the question, the already-fetched grounded
 * answer, and an explicit instruction not to alter it. The model's only job
 * on receiving this is to relay — not to fetch, not to reason, not to
 * re-answer from its own training.
 */
function buildRelayInstruction(question: string, answer: string): string {
  return [
    `The user asked the Permaculture Ethics & Principles reference agent: "${question}"`,
    '',
    'Its grounded answer, already fetched from the reference document, is below. ' +
      'Reply with this answer essentially verbatim — do not paraphrase, summarize, ' +
      'shorten, add commentary, or change its wording. Just present it as your reply:',
    '',
    answer,
  ].join('\n')
}

/** Outcome of a webhook call: either the trimmed grounded answer, or an error message to show the caller. */
type FetchOutcome =
  | { kind: 'answer'; answer: string }
  | { kind: 'error'; text: string }

/** Call the n8n webhook and interpret its response. Shared by the command and the follow-up tool. */
async function fetchGroundedAnswer(config: PermacultureCommandConfig, question: string): Promise<FetchOutcome> {
  let response: Response
  try {
    response = await fetch(config.webhookUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...config.headers },
      body: JSON.stringify({ chatInput: question }),
    })
  } catch (error) {
    return {
      kind: 'error',
      text: `Could not reach the Permaculture Ethics & Principles agent: ${error instanceof Error ? error.message : String(error)}`,
    }
  }

  // Read the body exactly once, then try to interpret it — the n8n Agent
  // node's output shape is `{ output: "<answer text>" }`, but fall back
  // gracefully if that ever changes.
  const raw = await response.text().catch(() => '')

  if (!response.ok) {
    return {
      kind: 'error',
      text: `Permaculture Ethics & Principles agent returned HTTP ${response.status}${raw ? `: ${raw.slice(0, 500)}` : ''}`,
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
    return { kind: 'error', text: 'The Permaculture Ethics & Principles agent returned an empty response.' }
  }

  return { kind: 'answer', answer }
}

/**
 * Push the already-fetched, already-correct answer into the agent's own turn
 * loop as a follow-up — the same mechanism `webhook` and `schedule` use for
 * external content — so it produces one real reply that renders like any
 * other chat message, with only a short collapsed summary in its own
 * message row.
 */
function relayAnswer(agent: Agent, question: string, answer: string): void {
  agent.followup(createUserMessage({
    content: [{ type: 'text', text: buildRelayInstruction(question, answer) }],
    source: {
      kind: 'plugin',
      plugin: 'permaculture-command',
      form: 'notice',
      summary: boundContextSummary(`The user asked the Permaculture Ethics & Principles reference agent: "${question}"`),
    },
  }))
}

const permacultureThreadStateSchema: ZodType<PermacultureThreadState> = zod.object({
  active: zod.boolean(),
}).strict()

/** Host-only projection of the logged `permaculture/thread` events. */
const permacultureThreadProjectionDefinition = {
  key: 'permacultureThread',
  stateVersion: 1,
  stateSchema: permacultureThreadStateSchema,
  init: () => ({ active: false }),
  apply: (state, event) => {
    if (event.type === 'permaculture/thread') {
      return { active: event.data.active }
    }
    return state
  },
} satisfies Omit<ProjectionDefinition<'permacultureThread', PermacultureThreadState>, 'wire'>

/** Guidance included only while a thread is open. */
function policySection(): string {
  return [
    'A Permaculture Ethics & Principles reference thread is open in this session (started by /permaculture-help).',
    `For on-topic follow-ups — including answering a clarifying question the reference agent asked — call the \`${ASK_PERMACULTURE_FOLLOWUP}\` tool with the follow-up as \`question\`, instead of answering yourself, declining, or asking the user to retype the command.`,
    'That tool relays the reference agent\'s grounded answer directly to the user as your next reply; do not also answer in this turn, and never paraphrase or summarize what it relays.',
    'This only covers continuing this thread — an unrelated, off-topic question still gets no special handling.',
    'The user can close this thread with /permaculture-help off.',
  ].join(' ')
}

const ASK_PERMACULTURE_FOLLOWUP_DESCRIPTION
  = 'Use only while a Permaculture Ethics & Principles thread is open. Ask the reference agent a follow-up question — '
  + 'including answering its own clarifying question — and relay its grounded answer to the user. '
  + 'Do not call this to start a brand-new, unrelated topic; use /permaculture-help for that.'

/**
 * `ctx.permacultureCommand`: registers `/permaculture-help`, folds the
 * logged thread state, includes the `permaculture:policy` section while a
 * thread is open, and registers the stable `ask_permaculture_followup` tool.
 */
export class PermacultureCommandController extends Service {
  static inject = ['tools', 'systemPrompt', 'sessionProjections']

  /** Validated deployment-owned config. */
  private readonly config: PermacultureCommandConfig

  constructor(ctx: Context, rawConfig: PermacultureCommandConfig) {
    super(ctx, 'permacultureCommand')
    this.config = resolveConfig(rawConfig)

    ctx.systemPrompt.section({
      name: 'permaculture:policy',
      order: 560,
      text: (context) => {
        if (context.agent === undefined) return ''
        return this.threadActive(context.agent.session) ? policySection() : ''
      },
    })

    ctx.sessionProjections.register(permacultureThreadProjectionDefinition)

    // The command child activates only when a command registry is composed,
    // matching dsh-plan-mode's own guard for the same reason.
    ctx.inject(['commands'], (commandCtx) => {
      commandCtx.commands.register({
        name: 'permaculture-help',
        description: 'Ask the Permaculture Ethics & Principles reference agent (grounded in the reference document, via n8n)',
        input: { hint: '<question>|off' },
        handler: async ({ agent, rawInput }) => {
          const input = rawInput.trim()

          if (input === 'off') {
            if (!this.threadActive(agent.session)) {
              return { kind: 'success', text: 'Permaculture thread is already closed.' }
            }
            agent.session.append('permaculture/thread', { active: false }, { ignorable: true })
            return { kind: 'success', text: 'Permaculture thread closed.' }
          }

          const question = input
          if (question === '') {
            return { kind: 'error', text: 'Usage: /permaculture-help <question>, or /permaculture-help off to close an open thread.' }
          }

          const outcome = await fetchGroundedAnswer(this.config, question)
          if (outcome.kind === 'error') {
            return { kind: 'error', text: outcome.text }
          }

          if (!this.threadActive(agent.session)) {
            agent.session.append('permaculture/thread', { active: true }, { ignorable: true })
          }

          relayAnswer(agent, question, outcome.answer)

          return {
            kind: 'success',
            text: 'Fetched the grounded answer — relaying it as a reply below. '
              + 'Follow-ups in this session go straight to the reference agent until you run /permaculture-help off.',
          }
        },
      })
    })

    ctx.tools.register(defineTool({
      name: ASK_PERMACULTURE_FOLLOWUP,
      description: ASK_PERMACULTURE_FOLLOWUP_DESCRIPTION,
      parameters: {
        question: { type: 'string', required: true, description: 'The on-topic follow-up question to ask the reference agent.' },
      },
      output: {
        schema: {
          type: 'object',
          additionalProperties: false,
          properties: {
            relayed: { type: 'boolean', const: true, required: true },
          },
        },
        render: () => [{
          type: 'text',
          text: 'Relayed the grounded answer to the user directly in a new reply — do not repeat, summarize, or paraphrase it yourself; your turn is complete.',
        }],
      },
      execute: async (args, exec) => {
        const agent = exec.agent
        if (agent === undefined) {
          throw new Error(`${ASK_PERMACULTURE_FOLLOWUP} requires a calling agent (no session to relay into)`)
        }
        if (!this.threadActive(agent.session)) {
          throw new Error(`${ASK_PERMACULTURE_FOLLOWUP} is only available while a Permaculture Ethics & Principles thread is open — start one with /permaculture-help <question>`)
        }
        const question = args.question.trim()
        if (question === '') {
          throw new Error(`${ASK_PERMACULTURE_FOLLOWUP} requires a non-empty question`)
        }
        const outcome = await fetchGroundedAnswer(this.config, question)
        if (outcome.kind === 'error') {
          throw new Error(outcome.text)
        }
        relayAnswer(agent, question, outcome.answer)
        // The relay above is queued as its own fresh turn; conclude this one
        // so the model does not also restate or paraphrase the answer here.
        exec.concludeTurn()
        return { relayed: true }
      },
    }))
  }

  /** Read the logged thread state, or fail at the first service access. */
  private threadActive(session: Session): boolean {
    const state = this.ctx.sessionProjections.stateOf(session, 'permacultureThread')
    if (state === undefined) throw new Error('permaculture-command requires the permacultureThread session projection')
    return state.active
  }
}

export default PermacultureCommandController
