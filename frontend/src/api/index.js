const BASE = '/api'

export async function analyzeQuery(query, options = {}) {
  const res = await fetch(`${BASE}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, channel: 'web', ...options }),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`)
  return res.json()
}

export async function educateQuery(question, options = {}) {
  const res = await fetch(`${BASE}/educate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, ...options }),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`)
  return res.json()
}

export async function checkHealth() {
  try {
    const res = await fetch(`${BASE}/health`, { signal: AbortSignal.timeout(4000) })
    if (!res.ok) return { andrew: 'error', bridge: 'error' }
    return res.json()
  } catch {
    return { andrew: 'offline', bridge: 'offline' }
  }
}
