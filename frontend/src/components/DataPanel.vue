<template>
  <div class="data-panel">

    <!-- ── Andrew data panel ────────────────────────────── -->
    <template v-if="store.isAndrew">
      <!-- Stat bar (only when data exists) -->
      <div v-if="hasData" class="stat-bar">
        <div class="stat">
          <span class="stat-label">Confidence</span>
          <div class="stat-value-wrap">
            <div class="confidence-track">
              <div class="confidence-fill" :style="{ width: (data.confidence * 100) + '%', '--c': confidenceColor }"></div>
            </div>
            <span class="stat-value" :style="{ color: confidenceColor }">
              {{ (data.confidence * 100).toFixed(0) }}%
            </span>
          </div>
        </div>
        <div class="stat">
          <span class="stat-label">Route</span>
          <span class="stat-value route-badge">{{ data.routing || '—' }}</span>
        </div>
        <div class="stat">
          <span class="stat-label">Cost</span>
          <span class="stat-value cost-val">${{ (data.costUsd || 0).toFixed(4) }}</span>
        </div>
        <div class="stat">
          <span class="stat-label">Time</span>
          <span class="stat-value">{{ data.elapsedSeconds }}s</span>
        </div>
      </div>

      <!-- HITL banner -->
      <div v-if="data.hitlRequired" class="hitl-banner">
        <span class="hitl-icon">⚠</span>
        <div>
          <strong>Human review required</strong>
          <span v-if="data.hitlReason" class="hitl-reason"> — {{ data.hitlReason }}</span>
        </div>
      </div>

      <!-- Tabs: Profile / Table / Chart / SQL -->
      <div v-if="hasData" class="panel-tabs">
        <button
          v-for="tab in panelTabs"
          :key="tab.id"
          class="panel-tab"
          :class="{ active: activeTab === tab.id }"
          @click="activeTab = tab.id"
        >
          <span>{{ tab.icon }}</span>
          {{ tab.label }}
          <span v-if="tab.id === 'table' && data.queryResults.length" class="count-badge">
            {{ data.queryResults.length }}
          </span>
          <span v-if="tab.id === 'profile' && profileTableCount" class="count-badge profile-badge">
            {{ profileTableCount }}
          </span>
        </button>
      </div>

      <!-- Tab content -->
      <div class="panel-body">
        <!-- Empty state -->
        <div v-if="!hasData" class="panel-empty">
          <div class="panel-empty-icon">📋</div>
          <h3>No data yet</h3>
          <p>Ask Andrew an analytical question — SQL results and charts will appear here.</p>
        </div>

        <template v-else>
          <!-- Profile tab (Phase 1: Explore Data) -->
          <div v-if="activeTab === 'profile'" class="tab-content tab-profile">
            <DataProfilePanel v-if="data.dataProfile" :profile="data.dataProfile" />
            <div v-else class="no-results">No data profile available for this query.</div>
          </div>

          <!-- Table tab -->
          <div v-if="activeTab === 'table'" class="tab-content">
            <SqlTable v-if="data.queryResults.length" :rows="data.queryResults" />
            <div v-else class="no-results">No tabular results for this query.</div>
          </div>

          <!-- Chart tab -->
          <div v-if="activeTab === 'chart'" class="tab-content">
            <ChartView v-if="data.queryResults.length" :rows="data.queryResults" />
            <div v-else class="no-results">No numeric data to chart.</div>
          </div>

          <!-- SQL tab -->
          <div v-if="activeTab === 'sql'" class="tab-content sql-tab">
            <div class="sql-label">Generated SQL</div>
            <pre v-if="data.sqlQuery" class="sql-code">{{ data.sqlQuery }}</pre>
            <div v-else class="no-results">No SQL query for this response.</div>
          </div>
        </template>
      </div>
    </template>

    <!-- ── Romeo info panel ──────────────────────────────── -->
    <template v-else>
      <div class="romeo-panel">
        <div class="romeo-header">
          <span class="romeo-avatar">🎓</span>
          <div>
            <h2 class="romeo-title">Romeo PhD</h2>
            <p class="romeo-sub">Educational AI Agent</p>
          </div>
        </div>

        <div class="romeo-topics">
          <h3 class="topics-title">Areas of expertise</h3>
          <div class="topics-grid">
            <div v-for="topic in topics" :key="topic.label" class="topic-card">
              <span class="topic-icon">{{ topic.icon }}</span>
              <span class="topic-label">{{ topic.label }}</span>
            </div>
          </div>
        </div>

        <div class="romeo-tips">
          <h3 class="topics-title">Tips for great answers</h3>
          <ul class="tips-list">
            <li v-for="tip in tips" :key="tip">{{ tip }}</li>
          </ul>
        </div>

        <div class="romeo-tagline">
          <span class="romeo-cta">Romeo explains — Andrew computes</span>
          <p>For data analysis and SQL queries, switch to the Andrew tab.</p>
        </div>
      </div>
    </template>

  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useAgentStore } from '../stores/agent.js'
import SqlTable from './SqlTable.vue'
import ChartView from './ChartView.vue'
import DataProfilePanel from './DataProfilePanel.vue'

const store = useAgentStore()
const data  = computed(() => store.latestData)
const hasData = computed(() =>
  data.value.narrative ||
  data.value.queryResults?.length ||
  data.value.sqlQuery ||
  data.value.dataProfile
)

const activeTab = ref('table')

const panelTabs = [
  { id: 'profile', icon: '🔍', label: 'Profile' },
  { id: 'table',   icon: '⬛', label: 'Table' },
  { id: 'chart',   icon: '📈', label: 'Chart' },
  { id: 'sql',     icon: '🗄',  label: 'SQL' },
]

const profileTableCount = computed(() =>
  data.value.dataProfile?.tables?.length || 0
)

const confidenceColor = computed(() => {
  const c = data.value.confidence || 0
  if (c >= 0.75) return 'var(--success)'
  if (c >= 0.50) return 'var(--warning)'
  return 'var(--error)'
})

const topics = [
  { icon: '📊', label: 'Data Science' },
  { icon: '🤖', label: 'Machine Learning' },
  { icon: '📉', label: 'Statistics' },
  { icon: '🧮', label: 'Mathematics' },
  { icon: '💻', label: 'Programming' },
  { icon: '🔢', label: 'Deep Learning' },
  { icon: '📦', label: 'Databases & SQL' },
  { icon: '🧠', label: 'AI & LLMs' },
]

const tips = [
  'Be specific — "explain p-value for a clinical trial" beats "explain p-value"',
  'Ask for analogies if the concept feels abstract',
  'Request code examples when learning algorithms',
  'Ask Romeo to suggest a learning path for a new topic',
]
</script>

<style scoped>
.data-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: var(--bg);
  min-height: 0;
  overflow: hidden;
}

/* ── Stat bar ──────────────────────────────────────────── */
.stat-bar {
  display: flex;
  gap: 0;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}
.stat {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 5px;
  padding: 14px 16px;
  border-right: 1px solid var(--border);
}
.stat:last-child { border-right: none; }
.stat-label {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-dim);
}
.stat-value {
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
  font-family: 'JetBrains Mono', monospace;
}
.stat-value-wrap { display: flex; align-items: center; gap: 8px; }
.confidence-track {
  flex: 1;
  height: 4px;
  border-radius: 99px;
  background: rgba(255,255,255,0.08);
  overflow: hidden;
}
.confidence-fill {
  height: 100%;
  border-radius: 99px;
  background: var(--c, var(--success));
  transition: width 600ms ease;
}
.route-badge {
  font-size: 11px;
  background: var(--andrew-dim);
  color: var(--andrew);
  padding: 2px 8px;
  border-radius: 99px;
  font-family: 'JetBrains Mono', monospace;
}
.cost-val { color: var(--warning); }

/* ── HITL banner ───────────────────────────────────────── */
.hitl-banner {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 16px;
  background: rgba(245,158,11,0.07);
  border-bottom: 1px solid rgba(245,158,11,0.25);
  font-size: 13px;
  color: var(--warning);
  flex-shrink: 0;
}
.hitl-icon   { font-size: 16px; }
.hitl-reason { color: var(--text-muted); font-weight: 400; }

/* ── Panel tabs ────────────────────────────────────────── */
.panel-tabs {
  display: flex;
  gap: 2px;
  padding: 10px 14px 0;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}
.panel-tab {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 7px 14px;
  border: none;
  border-bottom: 2px solid transparent;
  background: transparent;
  color: var(--text-muted);
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all var(--transition);
  font-family: inherit;
  margin-bottom: -1px;
}
.panel-tab:hover { color: var(--text); }
.panel-tab.active {
  color: var(--andrew);
  border-bottom-color: var(--andrew);
}
.count-badge {
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 99px;
  background: var(--andrew-dim);
  color: var(--andrew);
  font-weight: 600;
}
.profile-badge {
  background: rgba(0,240,255,0.06);
  color: var(--text-dim);
}

/* ── Panel body ────────────────────────────────────────── */
.panel-body {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
}

.panel-empty {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  color: var(--text-muted);
  text-align: center;
  padding: 40px 30px;
}
.panel-empty-icon { font-size: 48px; }
.panel-empty h3 { font-size: 16px; font-weight: 600; color: var(--text); }
.panel-empty p  { font-size: 13px; max-width: 300px; }

.tab-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
  padding: 14px;
  overflow: hidden;
}

.no-results {
  color: var(--text-dim);
  font-size: 13px;
  padding: 20px 0;
  text-align: center;
}

.tab-profile { padding: 0; overflow: hidden; }
.sql-tab { gap: 10px; }
.sql-label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-dim);
}
.sql-code {
  flex: 1;
  background: rgba(0,0,0,0.4);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  color: var(--andrew);
  white-space: pre-wrap;
  word-break: break-all;
  overflow: auto;
}

/* ── Romeo info panel ──────────────────────────────────── */
.romeo-panel {
  flex: 1;
  padding: 28px 28px;
  display: flex;
  flex-direction: column;
  gap: 28px;
  overflow-y: auto;
}

.romeo-header {
  display: flex;
  align-items: center;
  gap: 16px;
}
.romeo-avatar { font-size: 48px; line-height: 1; }
.romeo-title  { font-size: 22px; font-weight: 700; color: var(--romeo); }
.romeo-sub    { font-size: 13px; color: var(--text-muted); margin-top: 2px; }

.topics-title {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-dim);
  margin-bottom: 12px;
}

.topics-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
}
.topic-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  padding: 14px 8px;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  transition: all var(--transition);
}
.topic-card:hover {
  background: var(--card-hover);
  border-color: var(--romeo-glow);
}
.topic-icon  { font-size: 22px; }
.topic-label { font-size: 11px; color: var(--text-muted); font-weight: 500; text-align: center; }

.tips-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding-left: 16px;
  color: var(--text-muted);
  font-size: 13px;
}
.tips-list li::marker { color: var(--romeo); }

.romeo-tagline {
  padding: 16px 18px;
  background: var(--romeo-dim);
  border: 1px solid rgba(168,85,247,0.2);
  border-radius: var(--radius);
}
.romeo-cta {
  display: block;
  font-weight: 600;
  color: var(--romeo);
  margin-bottom: 4px;
}
.romeo-tagline p { font-size: 12px; color: var(--text-muted); }
</style>
