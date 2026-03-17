<template>
  <div class="bubble-wrap" :class="msg.role">
    <!-- User bubble -->
    <div v-if="msg.role === 'user'" class="bubble user-bubble">
      {{ msg.text }}
    </div>

    <!-- Assistant bubble -->
    <div v-else class="bubble assistant-bubble" :class="[`bubble-${msg.agent}`, { error: msg.isError }]">
      <!-- Markdown content for Romeo -->
      <div v-if="msg.markdown" class="md" v-html="renderedMarkdown"></div>
      <!-- Plain text for Andrew -->
      <div v-else class="plain-text">{{ msg.text }}</div>

      <!-- Metadata footer -->
      <div v-if="msg.meta && !msg.isError" class="meta-footer">
        <span v-if="msg.meta.confidence != null" class="meta-chip">
          {{ (msg.meta.confidence * 100).toFixed(0) }}% confidence
        </span>
        <span v-if="msg.meta.routing" class="meta-chip chip-route">
          {{ msg.meta.routing }}
        </span>
        <span v-if="msg.meta.model" class="meta-chip">
          {{ msg.meta.model }}
        </span>
        <span v-if="msg.meta.costUsd != null" class="meta-chip chip-cost">
          ${{ msg.meta.costUsd.toFixed(4) }}
        </span>
        <span v-if="msg.meta.elapsedSeconds != null" class="meta-chip chip-time">
          {{ msg.meta.elapsedSeconds }}s
        </span>
        <span class="meta-chip" :class="msg.meta.success ? 'chip-ok' : 'chip-err'">
          {{ msg.meta.success ? '✓' : '✗' }}
        </span>
        <span v-if="msg.meta.hitlRequired" class="meta-chip chip-hitl" :title="msg.meta.hitlReason">
          ⚠ Awaiting Review
        </span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { marked } from 'marked'

const props = defineProps({
  msg: { type: Object, required: true },
})

const renderedMarkdown = computed(() => {
  if (!props.msg.markdown) return ''
  try {
    return marked.parse(props.msg.text || '', { breaks: true, gfm: true })
  } catch {
    return props.msg.text || ''
  }
})
</script>

<style scoped>
.bubble-wrap {
  display: flex;
  margin-bottom: 12px;
  animation: fadeIn 220ms ease-out;
}
.bubble-wrap.user      { justify-content: flex-end; }
.bubble-wrap.assistant { justify-content: flex-start; }

.bubble {
  max-width: 88%;
  padding: 11px 15px;
  border-radius: var(--radius-lg);
  font-size: 14px;
  line-height: 1.6;
}

/* User */
.user-bubble {
  background: linear-gradient(135deg, #1e3a5f 0%, #1a2d4d 100%);
  border: 1px solid rgba(0, 212, 255, 0.2);
  color: var(--text);
  border-bottom-right-radius: 4px;
}

/* Assistant */
.assistant-bubble {
  background: var(--card);
  border: 1px solid var(--border);
  color: var(--text);
  border-bottom-left-radius: 4px;
}
.assistant-bubble.bubble-andrew { border-left: 2px solid var(--andrew); }
.assistant-bubble.bubble-romeo  { border-left: 2px solid var(--romeo); }
.assistant-bubble.error {
  border-left: 2px solid var(--error);
  color: var(--error);
}

.plain-text {
  white-space: pre-wrap;
  word-break: break-word;
}

/* Meta footer */
.meta-footer {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin-top: 10px;
  padding-top: 8px;
  border-top: 1px solid var(--border);
}
.meta-chip {
  font-size: 10px;
  font-weight: 500;
  padding: 2px 7px;
  border-radius: 99px;
  background: rgba(255,255,255,0.05);
  color: var(--text-dim);
  font-family: 'JetBrains Mono', monospace;
}
.chip-route { background: var(--andrew-dim); color: var(--andrew); }
.chip-cost  { color: var(--warning); }
.chip-time  { color: var(--text-muted); }
.chip-ok    { background: rgba(34,197,94,0.1);  color: var(--success); }
.chip-err   { background: rgba(239,68,68,0.1);  color: var(--error); }
.chip-hitl  { background: rgba(245,158,11,0.15); color: var(--warning); font-weight: 600; }

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}
</style>
