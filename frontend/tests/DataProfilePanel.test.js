/**
 * DataProfilePanel.vue — component unit tests
 */
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import DataProfilePanel from '../src/components/DataProfilePanel.vue'

function makeProfile(overrides = {}) {
  return {
    tables: [
      {
        name: 'sales',
        row_count: 12_500,
        quality_flags: [],
        columns: [
          {
            name: 'revenue',
            dtype: 'float',
            null_rate: 0.02,
            min_val: 10,
            max_val: 50000,
            avg_val: 1200.5,
            top_values: [],
            quality_flags: [],
          },
          {
            name: 'region',
            dtype: 'text',
            null_rate: 0.0,
            min_val: null,
            max_val: null,
            avg_val: null,
            top_values: ['North', 'South', 'East', 'West', 'Central'],
            quality_flags: [],
          },
        ],
      },
    ],
    warnings: [],
    error: null,
    ...overrides,
  }
}

describe('DataProfilePanel — rendering', () => {
  it('renders table name', () => {
    const wrapper = mount(DataProfilePanel, { props: { profile: makeProfile() } })
    expect(wrapper.text()).toContain('sales')
  })

  it('renders row count formatted', () => {
    const wrapper = mount(DataProfilePanel, { props: { profile: makeProfile() } })
    // 12500 → "12.5k"
    expect(wrapper.text()).toContain('12.5k')
  })

  it('columns are hidden by default (collapsed)', () => {
    const wrapper = mount(DataProfilePanel, { props: { profile: makeProfile() } })
    expect(wrapper.find('.col-grid').exists()).toBe(false)
  })

  it('expands to show columns on header click', async () => {
    const wrapper = mount(DataProfilePanel, { props: { profile: makeProfile() } })
    await wrapper.find('.table-header').trigger('click')
    expect(wrapper.find('.col-grid').exists()).toBe(true)
    expect(wrapper.text()).toContain('revenue')
    expect(wrapper.text()).toContain('region')
  })

  it('shows numeric MIN/AVG/MAX chips', async () => {
    const wrapper = mount(DataProfilePanel, { props: { profile: makeProfile() } })
    await wrapper.find('.table-header').trigger('click')
    const chips = wrapper.findAll('.stat-chip')
    const texts = chips.map(c => c.text())
    expect(texts.some(t => t.includes('min'))).toBe(true)
    expect(texts.some(t => t.includes('avg'))).toBe(true)
    expect(texts.some(t => t.includes('max'))).toBe(true)
  })

  it('shows top categorical values (max 3)', async () => {
    const wrapper = mount(DataProfilePanel, { props: { profile: makeProfile() } })
    await wrapper.find('.table-header').trigger('click')
    const valChips = wrapper.findAll('.val-chip')
    expect(valChips.length).toBeLessThanOrEqual(3)
    expect(valChips[0].text()).toBe('North')
  })

  it('collapses again on second click', async () => {
    const wrapper = mount(DataProfilePanel, { props: { profile: makeProfile() } })
    await wrapper.find('.table-header').trigger('click')
    await wrapper.find('.table-header').trigger('click')
    expect(wrapper.find('.col-grid').exists()).toBe(false)
  })
})

describe('DataProfilePanel — quality flags', () => {
  it('shows empty_table flag with error style', () => {
    const profile = makeProfile({
      tables: [{
        name: 'orders',
        row_count: 0,
        quality_flags: ['empty_table'],
        columns: [],
      }],
    })
    const wrapper = mount(DataProfilePanel, { props: { profile } })
    const badge = wrapper.find('.flag-error')
    expect(badge.exists()).toBe(true)
    expect(badge.text()).toBe('empty_table')
  })

  it('shows high_null_rate flag with warn style on column', async () => {
    const profile = makeProfile({
      tables: [{
        name: 'users',
        row_count: 1000,
        quality_flags: [],
        columns: [{
          name: 'phone',
          dtype: 'text',
          null_rate: 0.75,
          min_val: null, max_val: null, avg_val: null,
          top_values: [],
          quality_flags: ['high_null_rate'],
        }],
      }],
    })
    const wrapper = mount(DataProfilePanel, { props: { profile } })
    await wrapper.find('.table-header').trigger('click')
    const warnBadges = wrapper.findAll('.flag-warn')
    expect(warnBadges.length).toBeGreaterThan(0)
  })

  it('shows warnings banner when profile.warnings is non-empty', () => {
    const profile = makeProfile({ warnings: ['empty table: orders'] })
    const wrapper = mount(DataProfilePanel, { props: { profile } })
    expect(wrapper.find('.profile-warnings').exists()).toBe(true)
    expect(wrapper.text()).toContain('empty table: orders')
  })

  it('shows error banner when profile.error is set', () => {
    const profile = makeProfile({ tables: [], error: 'DB connection refused' })
    const wrapper = mount(DataProfilePanel, { props: { profile } })
    expect(wrapper.find('.profile-error').exists()).toBe(true)
    expect(wrapper.text()).toContain('DB connection refused')
  })
})

describe('DataProfilePanel — empty / edge cases', () => {
  it('shows no-profile message when tables is empty and no error', () => {
    const wrapper = mount(DataProfilePanel, {
      props: { profile: { tables: [], warnings: [], error: null } },
    })
    expect(wrapper.find('.no-profile').exists()).toBe(true)
  })

  it('formats large row count as M suffix', () => {
    const profile = makeProfile({
      tables: [{ name: 'events', row_count: 2_400_000, quality_flags: [], columns: [] }],
    })
    const wrapper = mount(DataProfilePanel, { props: { profile } })
    expect(wrapper.text()).toContain('2.4M')
  })
})
