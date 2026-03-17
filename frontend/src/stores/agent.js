import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { analyzeQuery, educateQuery, checkHealth } from '../api/index.js'

export const useAgentStore = defineStore('agent', () => {
  // ── State ─────────────────────────────────────────────────
  const activeAgent = ref('andrew')   // 'andrew' | 'romeo'
  const andrewMessages = ref([])
  const romeoMessages  = ref([])
  const loading = ref(false)
  const health  = ref({ andrew: 'checking', bridge: 'checking' })

  // Latest Andrew data for the data panel
  const latestData = ref({
    queryResults: [],
    sqlQuery:     '',
    narrative:    '',
    confidence:   null,
    costUsd:      null,
    routing:      '',
    elapsedSeconds: null,
    dataProfile:  null,   // DataProfile from Phase 1 (Explore Data)
  })

  // ── Getters ───────────────────────────────────────────────
  const messages = computed(() =>
    activeAgent.value === 'andrew' ? andrewMessages.value : romeoMessages.value
  )

  const isAndrew = computed(() => activeAgent.value === 'andrew')
  const isRomeo  = computed(() => activeAgent.value === 'romeo')

  // ── Actions ───────────────────────────────────────────────
  function setAgent(agent) {
    activeAgent.value = agent
  }

  async function sendMessage(text) {
    if (!text.trim() || loading.value) return

    const userMsg = { id: Date.now(), role: 'user', text: text.trim(), agent: activeAgent.value }

    if (activeAgent.value === 'andrew') {
      andrewMessages.value.push(userMsg)
    } else {
      romeoMessages.value.push(userMsg)
    }

    loading.value = true

    try {
      if (activeAgent.value === 'andrew') {
        const result = await analyzeQuery(text.trim())

        const botMsg = {
          id: Date.now() + 1,
          role: 'assistant',
          agent: 'andrew',
          text: result.narrative || result.error || 'No output.',
          meta: {
            confidence: result.confidence,
            costUsd: result.cost_usd,
            routing: result.routing,
            elapsedSeconds: result.elapsed_seconds,
            success: result.success,
          },
        }
        andrewMessages.value.push(botMsg)

        // Update data panel
        if (result.success) {
          latestData.value = {
            queryResults:   result.query_results || [],
            sqlQuery:       result.sql_query || '',
            narrative:      result.narrative || '',
            confidence:     result.confidence,
            costUsd:        result.cost_usd,
            routing:        result.routing,
            elapsedSeconds: result.elapsed_seconds,
            dataProfile:    result.data_profile || null,
          }
        }
      } else {
        const result = await educateQuery(text.trim())

        const botMsg = {
          id: Date.now() + 1,
          role: 'assistant',
          agent: 'romeo',
          text: result.answer || result.error || 'No answer.',
          meta: {
            costUsd: result.cost_usd,
            elapsedSeconds: result.elapsed_seconds,
            model: result.model,
            success: result.success,
          },
          markdown: true,
        }
        romeoMessages.value.push(botMsg)
      }
    } catch (err) {
      const errMsg = {
        id: Date.now() + 1,
        role: 'assistant',
        agent: activeAgent.value,
        text: `Error: ${err.message}`,
        isError: true,
        meta: { success: false },
      }
      if (activeAgent.value === 'andrew') {
        andrewMessages.value.push(errMsg)
      } else {
        romeoMessages.value.push(errMsg)
      }
    } finally {
      loading.value = false
    }
  }

  async function refreshHealth() {
    health.value = await checkHealth()
  }

  return {
    activeAgent,
    andrewMessages,
    romeoMessages,
    messages,
    loading,
    health,
    latestData,
    isAndrew,
    isRomeo,
    setAgent,
    sendMessage,
    refreshHealth,
  }
})
