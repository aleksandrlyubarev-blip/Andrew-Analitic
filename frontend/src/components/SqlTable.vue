<template>
  <div class="sql-table-wrap">
    <div class="table-scroll">
      <table class="sql-table">
        <thead>
          <tr>
            <th v-for="col in columns" :key="col">{{ col }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, i) in visibleRows" :key="i">
            <td v-for="col in columns" :key="col" :class="cellClass(row[col])">
              {{ format(row[col]) }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div v-if="rows.length > pageSize" class="table-footer">
      <span class="row-count">{{ rows.length }} rows</span>
      <div class="pagination">
        <button :disabled="page === 0" @click="page--">‹</button>
        <span>{{ page + 1 }} / {{ totalPages }}</span>
        <button :disabled="page >= totalPages - 1" @click="page++">›</button>
      </div>
    </div>
    <div v-else class="table-footer">
      <span class="row-count">{{ rows.length }} row{{ rows.length !== 1 ? 's' : '' }}</span>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  rows: { type: Array, default: () => [] },
})

const page     = ref(0)
const pageSize = 10

const columns = computed(() => {
  if (!props.rows.length) return []
  return Object.keys(props.rows[0])
})

const totalPages = computed(() => Math.ceil(props.rows.length / pageSize))

const visibleRows = computed(() =>
  props.rows.slice(page.value * pageSize, (page.value + 1) * pageSize)
)

function format(val) {
  if (val == null) return '—'
  if (typeof val === 'number') {
    return Number.isInteger(val) ? val.toLocaleString() : val.toLocaleString(undefined, { maximumFractionDigits: 2 })
  }
  return String(val)
}

function cellClass(val) {
  if (typeof val === 'number') return 'num'
  return ''
}
</script>

<style scoped>
.sql-table-wrap {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
}

.table-scroll {
  flex: 1;
  overflow: auto;
}

.sql-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}
.sql-table th {
  position: sticky;
  top: 0;
  background: var(--surface);
  padding: 9px 12px;
  text-align: left;
  font-weight: 600;
  font-size: 11px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--text-muted);
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
.sql-table td {
  padding: 8px 12px;
  border-bottom: 1px solid var(--border);
  color: var(--text);
  font-family: 'JetBrains Mono', monospace;
  white-space: nowrap;
}
.sql-table td.num { text-align: right; color: var(--andrew); }
.sql-table tr:hover td { background: rgba(255,255,255,0.03); }

.table-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  border-top: 1px solid var(--border);
  flex-shrink: 0;
}
.row-count { font-size: 11px; color: var(--text-dim); }

.pagination {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--text-muted);
}
.pagination button {
  width: 26px; height: 26px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--card);
  color: var(--text-muted);
  cursor: pointer;
  transition: all var(--transition);
  font-size: 14px;
  display: grid;
  place-items: center;
}
.pagination button:hover:not(:disabled) {
  background: var(--card-hover);
  color: var(--text);
}
.pagination button:disabled { opacity: 0.3; cursor: not-allowed; }
</style>
