<template>
  <div class="chart-wrap">
    <!-- Chart type selector -->
    <div class="chart-controls">
      <button
        v-for="type in chartTypes"
        :key="type.id"
        class="chart-type-btn"
        :class="{ active: activeType === type.id }"
        @click="activeType = type.id"
        :title="type.label"
      >
        {{ type.icon }}
      </button>
      <div class="axis-selectors">
        <select v-model="xKey" class="axis-select" title="X axis">
          <option v-for="col in columns" :key="col" :value="col">{{ col }}</option>
        </select>
        <span class="axis-sep">→</span>
        <select v-model="yKey" class="axis-select" title="Y axis (numeric)">
          <option v-for="col in numericColumns" :key="col" :value="col">{{ col }}</option>
        </select>
      </div>
    </div>

    <!-- Chart canvas -->
    <div class="canvas-wrap">
      <Bar v-if="activeType === 'bar'" :data="chartData" :options="chartOptions" />
      <Line v-else-if="activeType === 'line'" :data="chartData" :options="chartOptions" />
      <Doughnut v-else-if="activeType === 'donut'" :data="donutData" :options="donutOptions" />
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import {
  Chart as ChartJS,
  Title, Tooltip, Legend,
  BarElement, LineElement, PointElement, ArcElement,
  CategoryScale, LinearScale,
  Filler,
} from 'chart.js'
import { Bar, Line, Doughnut } from 'vue-chartjs'

ChartJS.register(
  Title, Tooltip, Legend,
  BarElement, LineElement, PointElement, ArcElement,
  CategoryScale, LinearScale,
  Filler,
)

const props = defineProps({
  rows: { type: Array, default: () => [] },
})

const chartTypes = [
  { id: 'bar',   icon: '▬', label: 'Bar chart' },
  { id: 'line',  icon: '〜', label: 'Line chart' },
  { id: 'donut', icon: '○', label: 'Donut chart' },
]
const activeType = ref('bar')

const columns = computed(() => props.rows.length ? Object.keys(props.rows[0]) : [])

const numericColumns = computed(() =>
  columns.value.filter(col => props.rows.some(r => typeof r[col] === 'number'))
)

const xKey = ref('')
const yKey = ref('')

watch(columns, (cols) => {
  if (!cols.length) return
  xKey.value = cols[0]
  const numCol = cols.find(c => props.rows.some(r => typeof r[c] === 'number'))
  yKey.value = numCol || cols[cols.length - 1]
}, { immediate: true })

const COLORS = [
  'rgba(0, 212, 255, 0.85)',
  'rgba(168, 85, 247, 0.85)',
  'rgba(34, 197, 94, 0.85)',
  'rgba(245, 158, 11, 0.85)',
  'rgba(239, 68, 68, 0.85)',
  'rgba(99, 102, 241, 0.85)',
]

const labels = computed(() => props.rows.map(r => String(r[xKey.value] ?? '')))
const values = computed(() => props.rows.map(r => Number(r[yKey.value]) || 0))

const chartData = computed(() => ({
  labels: labels.value,
  datasets: [{
    label: yKey.value,
    data: values.value,
    backgroundColor: activeType.value === 'bar'
      ? 'rgba(0, 212, 255, 0.7)'
      : 'rgba(0, 212, 255, 0.15)',
    borderColor: 'rgba(0, 212, 255, 1)',
    borderWidth: activeType.value === 'bar' ? 0 : 2,
    borderRadius: activeType.value === 'bar' ? 6 : 0,
    fill: activeType.value === 'line',
    tension: 0.4,
    pointBackgroundColor: 'rgba(0, 212, 255, 1)',
    pointRadius: 4,
    pointHoverRadius: 6,
  }],
}))

const donutData = computed(() => ({
  labels: labels.value,
  datasets: [{
    data: values.value,
    backgroundColor: COLORS,
    borderColor: 'rgba(7,7,20,0.9)',
    borderWidth: 3,
    hoverOffset: 8,
  }],
}))

const baseOptions = {
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 400 },
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: 'rgba(10,10,30,0.95)',
      titleColor: '#e2e8f0',
      bodyColor: '#94a3b8',
      borderColor: 'rgba(255,255,255,0.1)',
      borderWidth: 1,
      padding: 10,
    },
  },
}

const chartOptions = computed(() => ({
  ...baseOptions,
  scales: {
    x: {
      grid:  { color: 'rgba(255,255,255,0.04)' },
      ticks: { color: '#64748b', font: { size: 11 } },
    },
    y: {
      grid:  { color: 'rgba(255,255,255,0.04)' },
      ticks: { color: '#64748b', font: { size: 11 } },
    },
  },
}))

const donutOptions = {
  ...baseOptions,
  plugins: {
    ...baseOptions.plugins,
    legend: {
      display: true,
      position: 'right',
      labels: { color: '#94a3b8', font: { size: 12 }, boxWidth: 12, padding: 12 },
    },
  },
}
</script>

<style scoped>
.chart-wrap {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
  padding: 0 4px;
}

.chart-controls {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 0 12px;
  flex-shrink: 0;
}

.chart-type-btn {
  width: 32px; height: 32px;
  border: 1px solid var(--border);
  border-radius: 7px;
  background: var(--card);
  color: var(--text-muted);
  cursor: pointer;
  font-size: 14px;
  transition: all var(--transition);
  display: grid;
  place-items: center;
}
.chart-type-btn:hover  { background: var(--card-hover); color: var(--text); }
.chart-type-btn.active {
  background: var(--andrew-dim);
  border-color: var(--andrew);
  color: var(--andrew);
}

.axis-selectors {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-left: auto;
}
.axis-sep { color: var(--text-dim); font-size: 12px; }
.axis-select {
  padding: 4px 8px;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text-muted);
  font-size: 12px;
  cursor: pointer;
  outline: none;
  font-family: inherit;
}
.axis-select:focus { border-color: var(--border-mid); }

.canvas-wrap {
  flex: 1;
  position: relative;
  min-height: 0;
}
</style>
