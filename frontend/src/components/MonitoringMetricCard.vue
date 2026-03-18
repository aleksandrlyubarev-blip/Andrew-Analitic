<template>
  <article class="metric-card" :data-state="state">
    <div class="metric-header">
      <div>
        <p class="metric-label">{{ label }}</p>
        <h3 class="metric-value">{{ value }}</h3>
      </div>
      <span class="metric-badge">{{ badge }}</span>
    </div>
    <p class="metric-trend">{{ trend }}</p>
    <div class="metric-track" aria-hidden="true">
      <div class="metric-fill" :style="{ width: normalizedPercent + '%' }"></div>
    </div>
  </article>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  label: { type: String, required: true },
  value: { type: String, required: true },
  badge: { type: String, default: 'Nominal' },
  trend: { type: String, default: '' },
  percent: { type: Number, default: 0 },
  state: {
    type: String,
    default: 'normal',
    validator: (value) => ['normal', 'warning', 'critical'].includes(value),
  },
})

const normalizedPercent = computed(() => Math.max(0, Math.min(100, props.percent)))
</script>

<style scoped>
.metric-card {
  --card-bg: #1f2937;
  --card-border: rgba(148, 163, 184, 0.18);
  --accent: #94a3b8;
  --text-strong: #f8fafc;
  --text-muted: #cbd5e1;
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)), var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 14px;
  padding: 18px;
  color: var(--text-strong);
  box-shadow: 0 18px 48px rgba(2, 6, 23, 0.28);
}

.metric-card[data-state='normal'] {
  --accent: #94a3b8;
  --card-border: rgba(148, 163, 184, 0.22);
}

.metric-card[data-state='warning'] {
  --accent: #f59e0b;
  --card-border: rgba(245, 158, 11, 0.36);
}

.metric-card[data-state='critical'] {
  --accent: #ef4444;
  --card-border: rgba(239, 68, 68, 0.4);
}

.metric-header {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
}

.metric-label {
  margin: 0;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 11px;
  color: var(--text-muted);
}

.metric-value {
  margin: 8px 0 0;
  font-size: 34px;
  line-height: 1;
  font-weight: 700;
}

.metric-badge {
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  background: color-mix(in srgb, var(--accent) 14%, transparent);
  color: var(--accent);
  border: 1px solid color-mix(in srgb, var(--accent) 36%, transparent);
}

.metric-trend {
  margin: 14px 0 12px;
  color: var(--text-muted);
  font-size: 13px;
}

.metric-track {
  height: 8px;
  border-radius: 999px;
  background: rgba(148, 163, 184, 0.16);
  overflow: hidden;
}

.metric-fill {
  height: 100%;
  border-radius: inherit;
  background: var(--accent);
}
</style>
