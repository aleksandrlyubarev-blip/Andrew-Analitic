<template>
  <div class="chat-panel" :class="`panel-${store.activeAgent}`">
    <!-- Messages area -->
    <div ref="scrollEl" class="messages">
      <!-- Empty state -->
      <div v-if="store.messages.length === 0" class="empty-state">
        <div class="empty-avatar">{{ currentAgent.avatar }}</div>
        <h3 class="empty-name">{{ currentAgent.name }}</h3>
        <p class="empty-desc">{{ currentAgent.emptyText }}</p>
        <div class="example-prompts">
          <button
            v-for="prompt in currentAgent.examples"
            :key="prompt"
            class="example-btn"
            @click="sendExample(prompt)"
          >
            {{ prompt }}
          </button>
        </div>
      </div>

      <!-- Messages -->
      <MessageBubble v-for="msg in store.messages" :key="msg.id" :msg="msg" />

      <!-- Typing indicator -->
      <div v-if="store.loading" class="typing-indicator">
        <div class="typing-bubble" :class="`bubble-${store.activeAgent}`">
          <span></span><span></span><span></span>
        </div>
      </div>
    </div>

    <!-- Input bar -->
    <div class="input-area">
      <textarea
        ref="inputEl"
        v-model="inputText"
        class="chat-input"
        :placeholder="currentAgent.placeholder"
        :disabled="store.loading"
        rows="1"
        @keydown.enter.exact.prevent="submit"
        @input="autoResize"
      />
      <button
        class="send-btn"
        :class="`send-${store.activeAgent}`"
        :disabled="!inputText.trim() || store.loading"
        @click="submit"
        title="Send (Enter)"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <line x1="22" y1="2" x2="11" y2="13"/>
          <polygon points="22 2 15 22 11 13 2 9 22 2"/>
        </svg>
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick } from 'vue'
import { useAgentStore } from '../stores/agent.js'
import MessageBubble from './MessageBubble.vue'

const store    = useAgentStore()
const inputText = ref('')
const scrollEl  = ref(null)
const inputEl   = ref(null)

const agents = {
  andrew: {
    avatar: '📊',
    name: 'Andrew',
    emptyText: 'Your analytical AI. Ask me to analyse data, run SQL queries, create reports, and more.',
    placeholder: 'e.g. Total revenue by region this quarter…',
    examples: [
      'Total revenue by region',
      'Top 5 products by sales volume',
      'Monthly revenue trend',
      'Average order value by product',
    ],
  },
  romeo: {
    avatar: '🎓',
    name: 'Romeo PhD',
    emptyText: 'Your educational AI. Ask me to explain concepts in data science, ML, statistics, or programming.',
    placeholder: 'e.g. Explain the difference between precision and recall…',
    examples: [
      'What is gradient descent?',
      'Explain p-value in plain English',
      'How does a random forest work?',
      'What is overfitting and how to avoid it?',
    ],
  },
}

const currentAgent = computed(() => agents[store.activeAgent])

async function submit() {
  const text = inputText.value.trim()
  if (!text || store.loading) return
  inputText.value = ''
  await nextTick()
  resetHeight()
  store.sendMessage(text)
}

function sendExample(text) {
  inputText.value = text
  submit()
}

function autoResize() {
  const el = inputEl.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, 120) + 'px'
}

function resetHeight() {
  if (inputEl.value) inputEl.value.style.height = 'auto'
}

// Scroll to bottom on new messages
watch(
  () => store.messages.length + (store.loading ? 1 : 0),
  async () => {
    await nextTick()
    if (scrollEl.value) {
      scrollEl.value.scrollTop = scrollEl.value.scrollHeight
    }
  }
)
</script>

<style scoped>
.chat-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: var(--surface);
  border-top: 1px solid var(--border);
  min-height: 0;
  transition: border-color var(--transition);
}
.chat-panel.panel-andrew { border-top-color: var(--andrew); }
.chat-panel.panel-romeo  { border-top-color: var(--romeo); }

/* Messages */
.messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px 16px;
  display: flex;
  flex-direction: column;
}

/* Empty state */
.empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  gap: 10px;
  padding: 40px 20px;
  color: var(--text-muted);
}
.empty-avatar { font-size: 48px; line-height: 1; }
.empty-name   { font-size: 18px; font-weight: 700; color: var(--text); }
.empty-desc   { font-size: 13px; max-width: 280px; color: var(--text-muted); }

.example-prompts {
  display: flex;
  flex-direction: column;
  gap: 7px;
  width: 100%;
  max-width: 300px;
  margin-top: 8px;
}
.example-btn {
  padding: 8px 14px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: rgba(255,255,255,0.02);
  color: var(--text-muted);
  font-size: 12px;
  cursor: pointer;
  transition: all var(--transition);
  text-align: left;
}
.example-btn:hover {
  background: rgba(255,255,255,0.05);
  border-color: var(--border-mid);
  color: var(--text);
}

/* Typing indicator */
.typing-indicator { display: flex; margin-bottom: 12px; }
.typing-bubble {
  padding: 12px 16px;
  border-radius: var(--radius-lg);
  border-radius-bottom-left: 4px;
  background: var(--card);
  border: 1px solid var(--border);
  display: flex;
  gap: 5px;
  align-items: center;
}
.typing-bubble.bubble-andrew { border-left: 2px solid var(--andrew); }
.typing-bubble.bubble-romeo  { border-left: 2px solid var(--romeo); }

.typing-bubble span {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--text-dim);
  animation: bounce 1.2s ease-in-out infinite;
}
.typing-bubble span:nth-child(2) { animation-delay: 0.15s; }
.typing-bubble span:nth-child(3) { animation-delay: 0.30s; }

@keyframes bounce {
  0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
  40%           { transform: translateY(-5px); opacity: 1; }
}

/* Input area */
.input-area {
  display: flex;
  gap: 10px;
  align-items: flex-end;
  padding: 12px 16px;
  border-top: 1px solid var(--border);
  background: rgba(0,0,0,0.2);
  flex-shrink: 0;
}

.chat-input {
  flex: 1;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px 14px;
  font-family: inherit;
  font-size: 14px;
  color: var(--text);
  resize: none;
  min-height: 40px;
  max-height: 120px;
  overflow-y: auto;
  transition: border-color var(--transition);
  outline: none;
  line-height: 1.5;
}
.chat-input::placeholder { color: var(--text-dim); }
.chat-input:focus { border-color: var(--border-mid); }
.chat-input:disabled { opacity: 0.5; cursor: not-allowed; }

.send-btn {
  width: 40px; height: 40px;
  border-radius: 10px;
  border: none;
  cursor: pointer;
  display: grid;
  place-items: center;
  transition: all var(--transition);
  flex-shrink: 0;
  color: #000;
}
.send-btn:disabled { opacity: 0.35; cursor: not-allowed; }
.send-andrew { background: var(--andrew); }
.send-andrew:hover:not(:disabled) { background: #33ddff; box-shadow: 0 0 16px var(--andrew-glow); }
.send-romeo  { background: var(--romeo); }
.send-romeo:hover:not(:disabled)  { background: #bf7fff; box-shadow: 0 0 16px var(--romeo-glow); }
</style>
