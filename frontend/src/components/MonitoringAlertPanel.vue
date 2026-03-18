<template>
  <section class="alert-panel">
    <header class="alert-header">
      <h3>Alarm Summary</h3>
      <span class="alert-count">{{ alerts.length }} active</span>
    </header>
    <ul class="alert-list">
      <li
        v-for="alert in alerts"
        :key="alert.id"
        class="alert-item"
        :data-severity="alert.severity"
      >
        <div>
          <p class="alert-title">{{ alert.title }}</p>
          <p class="alert-meta">{{ alert.source }} · {{ alert.timestamp }}</p>
        </div>
        <span class="alert-severity">{{ alert.severity }}</span>
      </li>
    </ul>
  </section>
</template>

<script setup>
defineProps({
  alerts: {
    type: Array,
    default: () => [],
  },
})
</script>

<style scoped>
.alert-panel {
  background: #111827;
  color: #f8fafc;
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 14px;
  padding: 18px;
  box-shadow: 0 18px 48px rgba(2, 6, 23, 0.28);
}

.alert-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 14px;
}

.alert-header h3 {
  margin: 0;
  font-size: 16px;
}

.alert-count {
  color: #cbd5e1;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.alert-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: grid;
  gap: 10px;
}

.alert-item {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: center;
  padding: 12px 14px;
  border-radius: 12px;
  border: 1px solid rgba(148, 163, 184, 0.12);
  background: rgba(255, 255, 255, 0.02);
}

.alert-item[data-severity='warning'] {
  border-color: rgba(245, 158, 11, 0.36);
}

.alert-item[data-severity='critical'] {
  border-color: rgba(239, 68, 68, 0.42);
}

.alert-title {
  margin: 0;
  font-weight: 600;
}

.alert-meta {
  margin: 4px 0 0;
  color: #94a3b8;
  font-size: 12px;
}

.alert-severity {
  text-transform: uppercase;
  font-size: 11px;
  letter-spacing: 0.08em;
  color: #cbd5e1;
}
</style>
