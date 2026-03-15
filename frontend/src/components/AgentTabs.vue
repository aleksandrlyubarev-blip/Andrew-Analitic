<template>
  <div class="agent-tabs">
    <button
      v-for="agent in agents"
      :key="agent.id"
      class="tab"
      :class="[`tab-${agent.id}`, { active: store.activeAgent === agent.id }]"
      @click="store.setAgent(agent.id)"
    >
      <span class="tab-avatar">{{ agent.avatar }}</span>
      <div class="tab-text">
        <span class="tab-name">{{ agent.name }}</span>
        <span class="tab-role">{{ agent.role }}</span>
      </div>
      <span v-if="msgCount(agent.id)" class="tab-badge">{{ msgCount(agent.id) }}</span>
    </button>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useAgentStore } from '../stores/agent.js'

const store = useAgentStore()

const agents = [
  { id: 'andrew', avatar: '📊', name: 'Andrew', role: 'Analytics Agent' },
  { id: 'romeo',  avatar: '🎓', name: 'Romeo PhD', role: 'Educational Agent' },
]

function msgCount(id) {
  const msgs = id === 'andrew' ? store.andrewMessages : store.romeoMessages
  return msgs.filter(m => m.role === 'user').length || null
}
</script>

<style scoped>
.agent-tabs {
  display: flex;
  gap: 0;
  padding: 12px 12px 0;
  flex-shrink: 0;
}

.tab {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  border: 1px solid var(--border);
  border-bottom: none;
  border-radius: var(--radius) var(--radius) 0 0;
  background: transparent;
  cursor: pointer;
  transition: background var(--transition), border-color var(--transition);
  color: var(--text-muted);
  position: relative;
  overflow: hidden;
}

.tab + .tab { margin-left: 6px; }

.tab::before {
  content: '';
  position: absolute;
  inset: 0;
  opacity: 0;
  transition: opacity var(--transition);
}

.tab-andrew::before { background: linear-gradient(135deg, var(--andrew-dim) 0%, transparent 70%); }
.tab-romeo::before  { background: linear-gradient(135deg, var(--romeo-dim)  0%, transparent 70%); }

.tab.active { background: var(--card); color: var(--text); }
.tab.active::before { opacity: 1; }

.tab-andrew.active { border-color: var(--andrew); }
.tab-romeo.active  { border-color: var(--romeo); }

.tab-avatar { font-size: 20px; line-height: 1; flex-shrink: 0; }

.tab-text {
  display: flex;
  flex-direction: column;
  gap: 1px;
  text-align: left;
}
.tab-name { font-size: 13px; font-weight: 600; }
.tab-role { font-size: 11px; color: var(--text-dim); }

.tab-andrew.active .tab-name { color: var(--andrew); }
.tab-romeo.active  .tab-name { color: var(--romeo); }

.tab-badge {
  margin-left: auto;
  min-width: 20px;
  height: 20px;
  padding: 0 6px;
  border-radius: 99px;
  font-size: 11px;
  font-weight: 600;
  display: grid;
  place-items: center;
}
.tab-andrew .tab-badge { background: var(--andrew-dim); color: var(--andrew); }
.tab-romeo  .tab-badge { background: var(--romeo-dim);  color: var(--romeo); }
</style>
