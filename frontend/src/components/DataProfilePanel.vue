<template>
  <div class="profile-panel">

    <!-- Error banner -->
    <div v-if="profile.error" class="profile-error">
      <span class="err-icon">⚠</span>
      <span>{{ profile.error }}</span>
    </div>

    <!-- Warnings banner -->
    <div v-if="profile.warnings?.length" class="profile-warnings">
      <div class="warn-title">Quality warnings</div>
      <ul class="warn-list">
        <li v-for="w in profile.warnings" :key="w">{{ w }}</li>
      </ul>
    </div>

    <!-- Table cards -->
    <div v-if="profile.tables?.length" class="tables-list">
      <div
        v-for="table in profile.tables"
        :key="table.name"
        class="table-card"
      >
        <!-- Table header -->
        <div class="table-header" @click="toggle(table.name)">
          <div class="table-meta">
            <span class="table-name">{{ table.name }}</span>
            <span class="row-count">{{ fmtRows(table.row_count) }} rows</span>
            <span
              v-for="flag in table.quality_flags"
              :key="flag"
              class="flag-badge"
              :class="flagClass(flag)"
            >{{ flag }}</span>
          </div>
          <span class="chevron" :class="{ open: expanded.has(table.name) }">›</span>
        </div>

        <!-- Column detail (expandable) -->
        <div v-if="expanded.has(table.name)" class="col-grid">
          <div
            v-for="col in table.columns"
            :key="col.name"
            class="col-row"
            :class="{ 'col-flagged': col.quality_flags?.length }"
          >
            <div class="col-name">{{ col.name }}</div>
            <div class="col-dtype">{{ col.dtype }}</div>

            <!-- Null rate bar -->
            <div class="null-bar-wrap" title="Null rate">
              <div class="null-bar-track">
                <div
                  class="null-bar-fill"
                  :style="{ width: (col.null_rate * 100) + '%', '--nc': nullColor(col.null_rate) }"
                ></div>
              </div>
              <span class="null-pct" :style="{ color: nullColor(col.null_rate) }">
                {{ (col.null_rate * 100).toFixed(1) }}% null
              </span>
            </div>

            <!-- Numeric stats -->
            <div v-if="col.min_val != null" class="num-stats">
              <span class="stat-chip">min {{ fmt(col.min_val) }}</span>
              <span class="stat-chip">avg {{ fmt(col.avg_val) }}</span>
              <span class="stat-chip">max {{ fmt(col.max_val) }}</span>
            </div>

            <!-- Top categorical values -->
            <div v-else-if="col.top_values?.length" class="top-vals">
              <span v-for="v in col.top_values.slice(0, 3)" :key="v" class="val-chip">{{ v }}</span>
            </div>

            <!-- Quality flags -->
            <div v-if="col.quality_flags?.length" class="col-flags">
              <span
                v-for="flag in col.quality_flags"
                :key="flag"
                class="flag-badge flag-small"
                :class="flagClass(flag)"
              >{{ flag }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- No tables -->
    <div v-else-if="!profile.error" class="no-profile">
      <div class="no-profile-icon">🔍</div>
      <p>No tables profiled — check DATABASE_URL and schema_context configuration.</p>
    </div>

  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  profile: { type: Object, required: true },
})

const expanded = ref(new Set())

function toggle(name) {
  if (expanded.value.has(name)) {
    expanded.value.delete(name)
  } else {
    expanded.value.add(name)
  }
  // trigger reactivity
  expanded.value = new Set(expanded.value)
}

function fmtRows(n) {
  if (n == null) return '?'
  return n >= 1_000_000 ? (n / 1_000_000).toFixed(1) + 'M'
       : n >= 1_000     ? (n / 1_000).toFixed(1) + 'k'
       : String(n)
}

function fmt(v) {
  if (v == null) return '—'
  return Number.isInteger(v) ? v.toLocaleString() : v.toFixed(2)
}

function nullColor(rate) {
  if (rate >= 0.5) return 'var(--error)'
  if (rate >= 0.2) return 'var(--warning)'
  return 'var(--success)'
}

function flagClass(flag) {
  if (flag === 'empty_table')    return 'flag-error'
  if (flag === 'all_null')       return 'flag-error'
  if (flag === 'zero_variance')  return 'flag-warn'
  if (flag === 'high_null_rate') return 'flag-warn'
  return 'flag-info'
}
</script>

<style scoped>
.profile-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
  overflow-y: auto;
  padding: 14px;
}

/* ── Banners ───────────────────────────────────────────── */
.profile-error {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  background: rgba(239,68,68,0.08);
  border: 1px solid rgba(239,68,68,0.25);
  border-radius: 8px;
  font-size: 13px;
  color: var(--error);
}
.err-icon { font-size: 16px; }

.profile-warnings {
  padding: 10px 14px;
  background: rgba(245,158,11,0.07);
  border: 1px solid rgba(245,158,11,0.2);
  border-radius: 8px;
}
.warn-title {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--warning);
  margin-bottom: 6px;
}
.warn-list {
  margin: 0;
  padding-left: 16px;
  font-size: 12px;
  color: var(--text-muted);
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.warn-list li::marker { color: var(--warning); }

/* ── Table card ────────────────────────────────────────── */
.tables-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.table-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
}

.table-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 14px;
  cursor: pointer;
  user-select: none;
  transition: background var(--transition);
}
.table-header:hover { background: var(--card-hover); }

.table-meta {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.table-name {
  font-size: 13px;
  font-weight: 600;
  color: var(--andrew);
  font-family: 'JetBrains Mono', monospace;
}
.row-count {
  font-size: 12px;
  color: var(--text-muted);
}

.chevron {
  font-size: 16px;
  color: var(--text-dim);
  transition: transform var(--transition);
  transform: rotate(0deg);
}
.chevron.open { transform: rotate(90deg); }

/* ── Column grid ───────────────────────────────────────── */
.col-grid {
  border-top: 1px solid var(--border);
  display: flex;
  flex-direction: column;
}
.col-row {
  display: grid;
  grid-template-columns: 140px 70px 1fr auto;
  align-items: center;
  gap: 10px;
  padding: 8px 14px;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  font-size: 12px;
}
.col-row:last-child { border-bottom: none; }
.col-row.col-flagged { background: rgba(245,158,11,0.04); }

.col-name {
  font-family: 'JetBrains Mono', monospace;
  color: var(--text);
  font-size: 12px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.col-dtype {
  font-size: 11px;
  color: var(--text-dim);
  font-family: 'JetBrains Mono', monospace;
}

.null-bar-wrap {
  display: flex;
  align-items: center;
  gap: 8px;
}
.null-bar-track {
  width: 60px;
  height: 4px;
  border-radius: 99px;
  background: rgba(255,255,255,0.08);
  overflow: hidden;
  flex-shrink: 0;
}
.null-bar-fill {
  height: 100%;
  border-radius: 99px;
  background: var(--nc, var(--success));
  transition: width 400ms ease;
}
.null-pct {
  font-size: 11px;
  font-family: 'JetBrains Mono', monospace;
  white-space: nowrap;
}

.num-stats, .top-vals, .col-flags {
  display: flex;
  align-items: center;
  gap: 4px;
  flex-wrap: wrap;
  justify-content: flex-end;
}
.stat-chip {
  font-size: 10px;
  padding: 2px 6px;
  border-radius: 99px;
  background: rgba(0,240,255,0.08);
  color: var(--andrew);
  font-family: 'JetBrains Mono', monospace;
  white-space: nowrap;
}
.val-chip {
  font-size: 10px;
  padding: 2px 6px;
  border-radius: 99px;
  background: rgba(255,255,255,0.05);
  color: var(--text-muted);
  max-width: 80px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* ── Flags ─────────────────────────────────────────────── */
.flag-badge {
  font-size: 10px;
  padding: 2px 7px;
  border-radius: 99px;
  font-weight: 600;
  letter-spacing: 0.04em;
}
.flag-small { padding: 1px 5px; font-size: 9px; }
.flag-error { background: rgba(239,68,68,0.12); color: var(--error); }
.flag-warn  { background: rgba(245,158,11,0.12); color: var(--warning); }
.flag-info  { background: rgba(0,240,255,0.08);  color: var(--andrew); }

/* ── Empty state ───────────────────────────────────────── */
.no-profile {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 40px 20px;
  text-align: center;
  color: var(--text-muted);
}
.no-profile-icon { font-size: 36px; }
.no-profile p { font-size: 13px; max-width: 280px; }
</style>
