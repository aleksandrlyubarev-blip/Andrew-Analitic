import MonitoringAlertPanel from '../components/MonitoringAlertPanel.vue'

export default {
  title: 'Monitoring/Alert Panel',
  component: MonitoringAlertPanel,
}

export const ActiveAlerts = {
  args: {
    alerts: [
      {
        id: 'gpu-temp',
        title: 'GPU temperature exceeded warning band',
        source: 'Render node A3',
        timestamp: '2026-03-18 12:40 UTC',
        severity: 'warning',
      },
      {
        id: 'api-failures',
        title: 'Inference error budget depleted',
        source: 'OpenAI connector',
        timestamp: '2026-03-18 12:42 UTC',
        severity: 'critical',
      },
    ],
  },
}
