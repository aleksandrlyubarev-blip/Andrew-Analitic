<template>
  <header class="header">
    <div class="header-left">
      <div class="logo">
        <span class="logo-icon">🔬</span>
        <span class="logo-text">Andrew <span class="amp">&</span> Romeo</span>
      </div>
      <span class="header-subtitle">AI Analytics Suite</span>
    </div>

    <div class="header-right">
      <div class="health-indicator" :class="overallStatus">
        <span class="health-dot"></span>
        <span class="health-label">{{ statusLabel }}</span>
      </div>
    </div>
  </header>
</template>

<script setup>
import { computed } from 'vue'
import { useAgentStore } from '../stores/agent.js'

const store = useAgentStore()

const overallStatus = computed(() => {
  const a = store.health.andrew
  if (a === 'ok') return 'ok'
  if (a === 'checking') return 'checking'
  return 'offline'
})

const statusLabel = computed(() => {
  if (overallStatus.value === 'ok')       return 'Online'
  if (overallStatus.value === 'checking') return 'Connecting…'
  return 'Offline'
})
</script>

<style scoped>
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  height: 56px;
  border-bottom: 1px solid var(--border);
  background: rgba(7, 7, 20, 0.9);
  backdrop-filter: blur(20px);
  flex-shrink: 0;
  z-index: 10;
}

.header-left { display: flex; align-items: center; gap: 16px; }

.logo {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 700;
  font-size: 16px;
  letter-spacing: -0.01em;
}
.logo-icon { font-size: 20px; }
.logo-text { color: var(--text); }
.amp { color: var(--text-muted); font-weight: 400; }

.header-subtitle {
  font-size: 11px;
  color: var(--text-dim);
  font-weight: 500;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding-left: 16px;
  border-left: 1px solid var(--border);
}

/* Health */
.health-indicator {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 5px 12px;
  border-radius: 99px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.03);
}
.health-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
}
.health-label { font-size: 12px; font-weight: 500; color: var(--text-muted); }

.health-indicator.ok .health-dot    { background: var(--success); box-shadow: 0 0 6px var(--success); }
.health-indicator.offline .health-dot { background: var(--error); }
.health-indicator.checking .health-dot {
  background: var(--warning);
  animation: pulse 1.2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.3; }
}
</style>
