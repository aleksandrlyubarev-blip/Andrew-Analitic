/**
 * Pinia store — unit tests (no network, no LLM)
 * vi.mock replaces the api module so sendMessage never hits the network.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import { useAgentStore } from '../src/stores/agent.js'

// ── mock the API module ────────────────────────────────────────────────────────
vi.mock('../src/api/index.js', () => ({
  analyzeQuery: vi.fn(),
  educateQuery:  vi.fn(),
  checkHealth:   vi.fn().mockResolvedValue({ andrew: 'ok', bridge: 'ok' }),
}))

import { analyzeQuery, educateQuery } from '../src/api/index.js'

// ── helpers ────────────────────────────────────────────────────────────────────
function makeAnalyzeResult(overrides = {}) {
  return {
    success: true,
    narrative: 'Revenue is $1 M',
    sql_query: 'SELECT SUM(revenue) FROM sales',
    query_results: [{ revenue: 1_000_000 }],
    confidence: 0.88,
    cost_usd: 0.003,
    routing: 'sql',
    elapsed_seconds: 1.2,
    data_profile: {
      tables: [{ name: 'sales', row_count: 5000, columns: [], quality_flags: [] }],
      warnings: [],
      error: null,
    },
    hitl_required: false,
    hitl_reason: null,
    ...overrides,
  }
}

// ── tests ──────────────────────────────────────────────────────────────────────
describe('useAgentStore — initial state', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('defaults to andrew agent', () => {
    const store = useAgentStore()
    expect(store.activeAgent).toBe('andrew')
  })

  it('starts with empty message lists', () => {
    const store = useAgentStore()
    expect(store.andrewMessages).toHaveLength(0)
    expect(store.romeoMessages).toHaveLength(0)
  })

  it('latestData starts with null profile and no HITL', () => {
    const store = useAgentStore()
    expect(store.latestData.dataProfile).toBeNull()
    expect(store.latestData.hitlRequired).toBe(false)
    expect(store.latestData.hitlReason).toBeNull()
  })

  it('setAgent switches active agent', () => {
    const store = useAgentStore()
    store.setAgent('romeo')
    expect(store.activeAgent).toBe('romeo')
    expect(store.isRomeo).toBe(true)
    expect(store.isAndrew).toBe(false)
  })
})

describe('useAgentStore — sendMessage (Andrew)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  it('adds user + assistant messages on success', async () => {
    analyzeQuery.mockResolvedValue(makeAnalyzeResult())
    const store = useAgentStore()
    await store.sendMessage('Show revenue')
    expect(store.andrewMessages).toHaveLength(2)
    expect(store.andrewMessages[0].role).toBe('user')
    expect(store.andrewMessages[1].role).toBe('assistant')
  })

  it('populates latestData from successful response', async () => {
    analyzeQuery.mockResolvedValue(makeAnalyzeResult())
    const store = useAgentStore()
    await store.sendMessage('Show revenue')
    const d = store.latestData
    expect(d.confidence).toBe(0.88)
    expect(d.routing).toBe('sql')
    expect(d.dataProfile.tables).toHaveLength(1)
    expect(d.hitlRequired).toBe(false)
  })

  it('sets hitlRequired when backend returns hitl_required=true', async () => {
    analyzeQuery.mockResolvedValue(makeAnalyzeResult({
      hitl_required: true,
      hitl_reason: 'confidence below threshold',
    }))
    const store = useAgentStore()
    await store.sendMessage('Predict churn')
    expect(store.latestData.hitlRequired).toBe(true)
    expect(store.latestData.hitlReason).toBe('confidence below threshold')
    // HITL badge propagated to the message meta
    const botMsg = store.andrewMessages[1]
    expect(botMsg.meta.hitlRequired).toBe(true)
  })

  it('still updates latestData when data_profile present but success=false', async () => {
    analyzeQuery.mockResolvedValue(makeAnalyzeResult({
      success: false,
      data_profile: {
        tables: [{ name: 'orders', row_count: 0, columns: [], quality_flags: ['empty_table'] }],
        warnings: ['empty table: orders'],
        error: null,
      },
    }))
    const store = useAgentStore()
    await store.sendMessage('Analyse orders')
    expect(store.latestData.dataProfile).not.toBeNull()
    expect(store.latestData.dataProfile.tables[0].quality_flags).toContain('empty_table')
  })

  it('adds error message on API exception', async () => {
    analyzeQuery.mockRejectedValue(new Error('network timeout'))
    const store = useAgentStore()
    await store.sendMessage('crash test')
    const msgs = store.andrewMessages
    expect(msgs[msgs.length - 1].isError).toBe(true)
    expect(msgs[msgs.length - 1].text).toMatch(/network timeout/)
  })

  it('does nothing when text is blank', async () => {
    const store = useAgentStore()
    await store.sendMessage('   ')
    expect(analyzeQuery).not.toHaveBeenCalled()
  })
})

describe('useAgentStore — sendMessage (Romeo)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  it('sends to Romeo and pushes to romeoMessages', async () => {
    educateQuery.mockResolvedValue({
      success: true,
      answer: 'A p-value is…',
      cost_usd: 0.001,
      elapsed_seconds: 0.8,
      model: 'gpt-4o-mini',
    })
    const store = useAgentStore()
    store.setAgent('romeo')
    await store.sendMessage('Explain p-value')
    expect(store.romeoMessages).toHaveLength(2)
    expect(store.andrewMessages).toHaveLength(0)
    expect(store.romeoMessages[1].markdown).toBe(true)
  })
})
