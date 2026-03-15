<template>
  <div class="app">
    <Header />
    <main class="workspace">
      <div class="chat-side">
        <AgentTabs />
        <ChatPanel />
      </div>
      <div class="data-side">
        <DataPanel />
      </div>
    </main>
  </div>
</template>

<script setup>
import { onMounted } from 'vue'
import { useAgentStore } from './stores/agent.js'
import Header    from './components/Header.vue'
import AgentTabs from './components/AgentTabs.vue'
import ChatPanel from './components/ChatPanel.vue'
import DataPanel from './components/DataPanel.vue'

const store = useAgentStore()
onMounted(() => {
  store.refreshHealth()
  setInterval(() => store.refreshHealth(), 30_000)
})
</script>

<style>
/* ── Reset & base ─────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:          #070714;
  --surface:     #0f0f24;
  --card:        #1a1a35;
  --card-hover:  #202042;
  --border:      rgba(255,255,255,0.07);
  --border-mid:  rgba(255,255,255,0.12);

  --andrew:      #00d4ff;
  --andrew-dim:  rgba(0, 212, 255, 0.12);
  --andrew-glow: rgba(0, 212, 255, 0.25);
  --romeo:       #a855f7;
  --romeo-dim:   rgba(168, 85, 247, 0.12);
  --romeo-glow:  rgba(168, 85, 247, 0.25);

  --text:        #e2e8f0;
  --text-muted:  #94a3b8;
  --text-dim:    #64748b;

  --success:     #22c55e;
  --error:       #ef4444;
  --warning:     #f59e0b;

  --radius:      12px;
  --radius-lg:   18px;
  --transition:  200ms cubic-bezier(0.4, 0, 0.2, 1);

  font-family: 'Inter', system-ui, sans-serif;
  font-size: 14px;
  line-height: 1.6;
  color: var(--text);
}

html, body { height: 100%; background: var(--bg); overflow: hidden; }
#app      { height: 100%; display: flex; flex-direction: column; }

/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track  { background: transparent; }
::-webkit-scrollbar-thumb  { background: rgba(255,255,255,0.12); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.22); }

/* ── App layout ───────────────────────────────────────────── */
.app {
  height: 100%;
  display: flex;
  flex-direction: column;
  background: var(--bg);
}

.workspace {
  flex: 1;
  display: grid;
  grid-template-columns: 420px 1fr;
  min-height: 0;
}

.chat-side {
  display: flex;
  flex-direction: column;
  border-right: 1px solid var(--border);
  min-height: 0;
}

.data-side {
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
}

/* ── Utility ──────────────────────────────────────────────── */
.glass {
  background: rgba(255,255,255,0.03);
  backdrop-filter: blur(16px);
  border: 1px solid var(--border);
}

.tag {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 8px;
  border-radius: 99px;
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.02em;
}

.tag-andrew { background: var(--andrew-dim); color: var(--andrew); }
.tag-romeo  { background: var(--romeo-dim);  color: var(--romeo); }
.tag-success{ background: rgba(34,197,94,0.12); color: var(--success); }
.tag-error  { background: rgba(239,68,68,0.12);  color: var(--error); }

/* Markdown from Romeo ─────────────────────────────────────── */
.md h1, .md h2, .md h3 { margin: 1em 0 0.4em; font-weight: 600; line-height: 1.3; }
.md h2 { font-size: 1.1em; }
.md h3 { font-size: 1em; }
.md p  { margin-bottom: 0.7em; }
.md ul, .md ol { padding-left: 1.4em; margin-bottom: 0.7em; }
.md li { margin-bottom: 0.25em; }
.md code {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.88em;
  background: rgba(255,255,255,0.08);
  padding: 1px 5px;
  border-radius: 4px;
}
.md pre {
  background: rgba(0,0,0,0.35);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px;
  overflow-x: auto;
  margin-bottom: 0.8em;
}
.md pre code { background: none; padding: 0; }
.md blockquote {
  border-left: 3px solid var(--romeo);
  padding-left: 12px;
  color: var(--text-muted);
  margin-bottom: 0.7em;
}
.md strong { color: #f1f5f9; }
</style>
