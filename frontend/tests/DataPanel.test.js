/**
 * DataPanel.vue — tab switching + HITL banner tests
 * Uses a stub store so no Pinia wiring needed.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia, defineStore } from 'pinia'

// Stub child components to keep tests fast
vi.mock('../src/components/SqlTable.vue',         () => ({ default: { template: '<div class="sql-table-stub"/>' } }))
vi.mock('../src/components/ChartView.vue',        () => ({ default: { template: '<div class="chart-stub"/>' } }))
vi.mock('../src/components/DataProfilePanel.vue', () => ({ default: { template: '<div class="profile-stub"/>', props: ['profile'] } }))

import DataPanel from '../src/components/DataPanel.vue'
import { useAgentStore } from '../src/stores/agent.js'

// Mock the api module to avoid network
vi.mock('../src/api/index.js', () => ({
  analyzeQuery: vi.fn(),
  educateQuery:  vi.fn(),
  checkHealth:   vi.fn().mockResolvedValue({ andrew: 'ok', bridge: 'ok' }),
}))

function makeLatestData(overrides = {}) {
  return {
    queryResults:  [{ col: 1 }],
    sqlQuery:      'SELECT 1',
    narrative:     'Here is the result',
    confidence:    0.9,
    costUsd:       0.002,
    routing:       'sql',
    elapsedSeconds: 1.0,
    dataProfile:   null,
    hitlRequired:  false,
    hitlReason:    null,
    ...overrides,
  }
}

describe('DataPanel — Andrew mode tabs', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('shows all four tabs when hasData is true', () => {
    const store = useAgentStore()
    store.latestData = makeLatestData()
    const wrapper = mount(DataPanel)
    const tabs = wrapper.findAll('.panel-tab')
    const labels = tabs.map(t => t.text())
    expect(labels.some(l => l.includes('Profile'))).toBe(true)
    expect(labels.some(l => l.includes('Table'))).toBe(true)
    expect(labels.some(l => l.includes('Chart'))).toBe(true)
    expect(labels.some(l => l.includes('SQL'))).toBe(true)
  })

  it('shows empty state when no data and no profile', () => {
    const store = useAgentStore()
    store.latestData = makeLatestData({
      queryResults: [], sqlQuery: '', narrative: '', dataProfile: null,
    })
    const wrapper = mount(DataPanel)
    expect(wrapper.find('.panel-empty').exists()).toBe(true)
  })

  it('shows Profile tab content when profile tab is clicked', async () => {
    const store = useAgentStore()
    store.latestData = makeLatestData({
      dataProfile: { tables: [{ name: 'sales', row_count: 100, columns: [], quality_flags: [] }], warnings: [], error: null },
    })
    const wrapper = mount(DataPanel)
    const profileTab = wrapper.findAll('.panel-tab').find(t => t.text().includes('Profile'))
    await profileTab.trigger('click')
    expect(wrapper.find('.profile-stub').exists()).toBe(true)
  })

  it('shows table count badge on Table tab', () => {
    const store = useAgentStore()
    store.latestData = makeLatestData({ queryResults: [{ a: 1 }, { a: 2 }, { a: 3 }] })
    const wrapper = mount(DataPanel)
    const badge = wrapper.find('.count-badge:not(.profile-badge)')
    expect(badge.text()).toBe('3')
  })

  it('shows profile table count badge on Profile tab when profile has tables', () => {
    const store = useAgentStore()
    store.latestData = makeLatestData({
      dataProfile: {
        tables: [
          { name: 'sales', row_count: 1000, columns: [], quality_flags: [] },
          { name: 'users', row_count: 500,  columns: [], quality_flags: [] },
        ],
        warnings: [],
        error: null,
      },
    })
    const wrapper = mount(DataPanel)
    const badge = wrapper.find('.profile-badge')
    expect(badge.text()).toBe('2')
  })
})

describe('DataPanel — HITL banner', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('does NOT show HITL banner when hitlRequired is false', () => {
    const store = useAgentStore()
    store.latestData = makeLatestData({ hitlRequired: false })
    const wrapper = mount(DataPanel)
    expect(wrapper.find('.hitl-banner').exists()).toBe(false)
  })

  it('shows HITL banner when hitlRequired is true', () => {
    const store = useAgentStore()
    store.latestData = makeLatestData({ hitlRequired: true, hitlReason: 'low confidence' })
    const wrapper = mount(DataPanel)
    expect(wrapper.find('.hitl-banner').exists()).toBe(true)
    expect(wrapper.text()).toContain('Human review required')
    expect(wrapper.text()).toContain('low confidence')
  })

  it('shows HITL banner without reason when hitlReason is null', () => {
    const store = useAgentStore()
    store.latestData = makeLatestData({ hitlRequired: true, hitlReason: null })
    const wrapper = mount(DataPanel)
    const banner = wrapper.find('.hitl-banner')
    expect(banner.exists()).toBe(true)
    expect(wrapper.find('.hitl-reason').exists()).toBe(false)
  })
})

describe('DataPanel — DataProfile shown on partial failure', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('shows panel (not empty state) when only dataProfile is set', () => {
    const store = useAgentStore()
    store.latestData = makeLatestData({
      queryResults: [],
      sqlQuery:     '',
      narrative:    '',
      dataProfile:  { tables: [{ name: 't', row_count: 0, columns: [], quality_flags: ['empty_table'] }], warnings: [], error: null },
    })
    const wrapper = mount(DataPanel)
    expect(wrapper.find('.panel-empty').exists()).toBe(false)
    expect(wrapper.find('.panel-tabs').exists()).toBe(true)
  })
})
