import MonitoringMetricCard from '../components/MonitoringMetricCard.vue'

export default {
  title: 'Monitoring/Metric Card',
  component: MonitoringMetricCard,
  args: {
    label: 'CPU Load',
    value: '42%',
    badge: 'Nominal',
    trend: 'Gray in steady-state, signal colors only on deviation.',
    percent: 42,
    state: 'normal',
  },
}

export const Normal = {}

export const Warning = {
  args: {
    label: 'GPU Saturation',
    value: '78%',
    badge: 'Watch',
    trend: 'Thermal headroom is shrinking in the last 10 minutes.',
    percent: 78,
    state: 'warning',
  },
}

export const Critical = {
  args: {
    label: 'Error Rate',
    value: '12.4%',
    badge: 'Alarm',
    trend: 'Escalate to operator. ISA-101 alarm palette engaged.',
    percent: 92,
    state: 'critical',
  },
}
