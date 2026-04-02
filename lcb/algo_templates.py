"""
lcb/algo_templates.py
=====================
RAG-style algorithm template store for hard competitive programming problems.

Each template provides:
  - A named algorithm pattern
  - Retrieval keywords
  - Python code scaffold with key invariants commented

Usage:
    from lcb.algo_templates import retrieve_templates
    templates = retrieve_templates(problem_statement, top_k=2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# ── Template definitions ──────────────────────────────────────


@dataclass
class AlgoTemplate:
    name: str
    keywords: List[str]       # retrieval triggers
    description: str
    code: str


# fmt: off
_TEMPLATES: List[AlgoTemplate] = [

    AlgoTemplate(
        name="Binary Search on Answer",
        keywords=["binary search", "minimize", "maximize", "smallest", "largest",
                  "at least", "at most", "threshold", "feasible"],
        description="Parametric binary search: check feasibility of answer value.",
        code='''\
def binary_search_answer(lo: int, hi: int) -> int:
    """Binary search on the answer space [lo, hi]."""
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(mid):   # define: is answer <= mid achievable?
            hi = mid
        else:
            lo = mid + 1
    return lo

def feasible(mid: int) -> bool:
    # TODO: implement feasibility check in O(n) or O(n log n)
    raise NotImplementedError
''',
    ),

    AlgoTemplate(
        name="Prefix Sum / Difference Array",
        keywords=["subarray sum", "range sum", "prefix", "difference array",
                  "range update", "point query", "cumulative"],
        description="O(1) range queries / range updates with prefix sums.",
        code='''\
# Prefix sum for range sum queries
prefix = [0] * (n + 1)
for i, v in enumerate(arr):
    prefix[i + 1] = prefix[i] + v

def range_sum(l: int, r: int) -> int:
    """Sum of arr[l..r] inclusive, 0-indexed."""
    return prefix[r + 1] - prefix[l]

# Difference array for range add, then prefix-sum to reconstruct
diff = [0] * (n + 1)
def range_add(l: int, r: int, val: int) -> None:
    diff[l] += val
    if r + 1 <= n:
        diff[r + 1] -= val

result = list(itertools.accumulate(diff))
''',
    ),

    AlgoTemplate(
        name="Monotonic Stack",
        keywords=["next greater", "next smaller", "previous greater", "previous smaller",
                  "stack", "monotonic", "histogram", "largest rectangle"],
        description="O(n) next-greater / next-smaller element via monotonic stack.",
        code='''\
def next_greater(arr: list[int]) -> list[int]:
    """For each index, index of next strictly greater element, or -1."""
    n = len(arr)
    result = [-1] * n
    stack: list[int] = []   # indices, values decrease
    for i, v in enumerate(arr):
        while stack and arr[stack[-1]] < v:
            result[stack.pop()] = i
        stack.append(i)
    return result
''',
    ),

    AlgoTemplate(
        name="Two Pointers / Sliding Window",
        keywords=["two pointers", "sliding window", "subarray", "substring",
                  "at most k", "exactly k", "contiguous", "window"],
        description="O(n) window expansion/contraction for subarray/substring problems.",
        code='''\
def sliding_window(arr: list[int], k: int) -> int:
    """Maximum sum subarray of length <= k (template)."""
    left = 0
    current = 0
    best = 0
    for right, v in enumerate(arr):
        current += v
        # Shrink window while constraint violated
        while current > k and left <= right:
            current -= arr[left]
            left += 1
        best = max(best, right - left + 1)  # or current, depending on goal
    return best
''',
    ),

    AlgoTemplate(
        name="Dynamic Programming — 1D",
        keywords=["dp", "dynamic programming", "subsequence", "subarray dp",
                  "coin change", "house robber", "fibonacci", "climb stairs"],
        description="Standard 1-D DP with rolling array optimisation.",
        code='''\
# Coin change (min coins to reach target)
def coin_change(coins: list[int], amount: int) -> int:
    INF = float("inf")
    dp = [INF] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for c in coins:
            if c <= i:
                dp[i] = min(dp[i], dp[i - c] + 1)
    return dp[amount] if dp[amount] < INF else -1

# LIS (Longest Increasing Subsequence) in O(n log n)
import bisect
def lis(nums: list[int]) -> int:
    tails: list[int] = []
    for x in nums:
        pos = bisect.bisect_left(tails, x)
        if pos == len(tails):
            tails.append(x)
        else:
            tails[pos] = x
    return len(tails)
''',
    ),

    AlgoTemplate(
        name="Dynamic Programming — 2D / Grid",
        keywords=["grid", "matrix dp", "path", "unique paths", "dungeon",
                  "2d dp", "rectangle", "interval dp", "dp on grid"],
        description="2-D DP for grid paths and interval/palindrome problems.",
        code='''\
# Unique paths in m×n grid
def unique_paths(m: int, n: int) -> int:
    dp = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[m-1][n-1]

# Interval DP template (e.g. matrix chain multiplication)
# dp[i][j] = optimal cost for subproblem [i..j]
dp = [[0] * n for _ in range(n)]
for length in range(2, n + 1):        # subproblem length
    for i in range(n - length + 1):
        j = i + length - 1
        dp[i][j] = min(
            dp[i][k] + dp[k+1][j] + cost(i, k, j)
            for k in range(i, j)
        )
''',
    ),

    AlgoTemplate(
        name="BFS / BFS on Grid",
        keywords=["bfs", "breadth first", "shortest path unweighted", "level order",
                  "minimum steps", "grid bfs", "0-1 bfs"],
        description="BFS shortest path on unweighted graph / grid.",
        code='''\
from collections import deque

def bfs(graph: dict, start: int, target: int) -> int:
    """Shortest path length (edges) from start to target. Returns -1 if unreachable."""
    dist = {start: 0}
    q = deque([start])
    while q:
        node = q.popleft()
        if node == target:
            return dist[node]
        for nb in graph.get(node, []):
            if nb not in dist:
                dist[nb] = dist[node] + 1
                q.append(nb)
    return -1

# 0-1 BFS (edge weights 0 or 1) — use deque, push front for 0-weight
def bfs_01(adj, src, n):
    dist = [float("inf")] * n
    dist[src] = 0
    dq = deque([src])
    while dq:
        u = dq.popleft()
        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                dq.appendleft(v) if w == 0 else dq.append(v)
    return dist
''',
    ),

    AlgoTemplate(
        name="Dijkstra's Algorithm",
        keywords=["dijkstra", "shortest path", "weighted graph", "minimum cost path",
                  "non-negative weights"],
        description="Dijkstra single-source shortest path with heapq.",
        code='''\
import heapq

def dijkstra(adj: list[list[tuple[int, int]]], src: int) -> list[float]:
    """adj[u] = [(v, weight), ...]. Returns dist[] from src."""
    n = len(adj)
    dist = [float("inf")] * n
    dist[src] = 0
    heap = [(0, src)]   # (dist, node)
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist
''',
    ),

    AlgoTemplate(
        name="Union-Find (Disjoint Set Union)",
        keywords=["union find", "dsu", "disjoint set", "connected components",
                  "cycle detection undirected", "kruskal", "spanning tree"],
        description="Path-compressed, rank-unioned DSU in near-O(1) per operation.",
        code='''\
class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # path halving
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        """Returns True if x and y were in different components."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.components -= 1
        return True
''',
    ),

    AlgoTemplate(
        name="Fenwick Tree (BIT)",
        keywords=["fenwick", "bit", "binary indexed tree", "prefix sum update",
                  "point update range query", "range update point query"],
        description="Fenwick tree for O(log n) prefix sum with point updates.",
        code='''\
class FenwickTree:
    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n + 1)

    def update(self, i: int, delta: int) -> None:
        """Add delta to position i (1-indexed)."""
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def query(self, i: int) -> int:
        """Prefix sum [1..i] (1-indexed)."""
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s

    def range_query(self, l: int, r: int) -> int:
        return self.query(r) - self.query(l - 1)
''',
    ),

    AlgoTemplate(
        name="Segment Tree",
        keywords=["segment tree", "range query", "range update", "lazy propagation",
                  "range minimum", "range maximum", "range sum update"],
        description="Iterative segment tree with lazy propagation for range updates.",
        code='''\
class SegTree:
    """Range sum query, range add update. Adapt for other monoids."""
    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (4 * n)
        self.lazy = [0] * (4 * n)

    def _push(self, node: int, l: int, r: int) -> None:
        if self.lazy[node]:
            mid = (l + r) // 2
            self._apply(2*node,   l,   mid, self.lazy[node])
            self._apply(2*node+1, mid+1, r, self.lazy[node])
            self.lazy[node] = 0

    def _apply(self, node: int, l: int, r: int, val: int) -> None:
        self.tree[node] += val * (r - l + 1)
        self.lazy[node] += val

    def update(self, node: int, l: int, r: int, ql: int, qr: int, val: int) -> None:
        if qr < l or r < ql:
            return
        if ql <= l and r <= qr:
            self._apply(node, l, r, val)
            return
        self._push(node, l, r)
        mid = (l + r) // 2
        self.update(2*node,   l,   mid, ql, qr, val)
        self.update(2*node+1, mid+1, r, ql, qr, val)
        self.tree[node] = self.tree[2*node] + self.tree[2*node+1]

    def query(self, node: int, l: int, r: int, ql: int, qr: int) -> int:
        if qr < l or r < ql:
            return 0
        if ql <= l and r <= qr:
            return self.tree[node]
        self._push(node, l, r)
        mid = (l + r) // 2
        return (self.query(2*node,   l,   mid, ql, qr) +
                self.query(2*node+1, mid+1, r, ql, qr))
''',
    ),

    AlgoTemplate(
        name="Topological Sort (Kahn's Algorithm)",
        keywords=["topological sort", "topological order", "dag", "directed acyclic",
                  "dependency", "course schedule", "prerequisites"],
        description="Kahn's BFS-based topological sort with cycle detection.",
        code='''\
from collections import deque

def topo_sort(n: int, edges: list[tuple[int, int]]) -> list[int] | None:
    """
    Returns topological order of nodes [0..n-1], or None if cycle detected.
    edges: list of (u, v) meaning u → v.
    """
    indegree = [0] * n
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        indegree[v] += 1

    q = deque(i for i in range(n) if indegree[i] == 0)
    order: list[int] = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                q.append(v)

    return order if len(order) == n else None   # None → cycle
''',
    ),

    AlgoTemplate(
        name="Matrix Exponentiation",
        keywords=["matrix exponentiation", "matrix power", "linear recurrence",
                  "fibonacci fast", "nth term", "recurrence relation"],
        description="O(k^3 log n) linear recurrence via matrix fast exponentiation.",
        code='''\
def mat_mul(A: list[list[int]], B: list[list[int]], MOD: int) -> list[list[int]]:
    n = len(A)
    C = [[0]*n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            if A[i][k] == 0:
                continue
            for j in range(n):
                C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % MOD
    return C

def mat_pow(M: list[list[int]], p: int, MOD: int) -> list[list[int]]:
    n = len(M)
    result = [[1 if i==j else 0 for j in range(n)] for i in range(n)]  # identity
    while p:
        if p & 1:
            result = mat_mul(result, M, MOD)
        M = mat_mul(M, M, MOD)
        p >>= 1
    return result

# Fibonacci example: [F(n+1), F(n)] = [[1,1],[1,0]]^n * [1, 0]
''',
    ),

    AlgoTemplate(
        name="Number Theory Utilities",
        keywords=["prime", "sieve", "gcd", "lcm", "modular inverse", "totient",
                  "factorization", "coprime", "euler"],
        description="Sieve of Eratosthenes, modular inverse, Euler totient.",
        code='''\
# Sieve of Eratosthenes
def sieve(limit: int) -> list[bool]:
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit+1, i):
                is_prime[j] = False
    return is_prime

# Modular inverse via Fermat's little theorem (MOD must be prime)
def modinv(a: int, MOD: int) -> int:
    return pow(a, MOD - 2, MOD)

# Extended GCD: returns (g, x, y) s.t. a*x + b*y = g
def ext_gcd(a: int, b: int) -> tuple[int, int, int]:
    if b == 0:
        return a, 1, 0
    g, x, y = ext_gcd(b, a % b)
    return g, y, x - (a // b) * y

# Euler's totient phi(n) in O(sqrt(n))
def euler_phi(n: int) -> int:
    result = n
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1
    if n > 1:
        result -= result // n
    return result
''',
    ),

    AlgoTemplate(
        name="Trie (Prefix Tree)",
        keywords=["trie", "prefix tree", "prefix", "autocomplete", "word search",
                  "xor maximum", "bitwise trie"],
        description="Array-based trie for prefix queries and XOR maximisation.",
        code='''\
class Trie:
    def __init__(self):
        self.children: dict[str, "Trie"] = {}
        self.is_end = False

    def insert(self, word: str) -> None:
        node = self
        for ch in word:
            node = node.children.setdefault(ch, Trie())
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_end

    def starts_with(self, prefix: str) -> bool:
        node = self
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True

# Binary trie for maximum XOR pair in O(n * 30)
class BinaryTrie:
    def __init__(self):
        self.ch = [None, None]

    def insert(self, num: int, bits: int = 30) -> None:
        node = self
        for i in range(bits, -1, -1):
            b = (num >> i) & 1
            if node.ch[b] is None:
                node.ch[b] = BinaryTrie()
            node = node.ch[b]

    def max_xor(self, num: int, bits: int = 30) -> int:
        node = self
        xor = 0
        for i in range(bits, -1, -1):
            b = (num >> i) & 1
            want = 1 - b
            if node.ch[want] is not None:
                xor |= (1 << i)
                node = node.ch[want]
            elif node.ch[b] is not None:
                node = node.ch[b]
            else:
                break
        return xor
''',
    ),
]
# fmt: on


# ── Retrieval ─────────────────────────────────────────────────

def retrieve_templates(problem_statement: str, top_k: int = 2) -> List[AlgoTemplate]:
    """
    Keyword-match problem statement against template retrieval keywords.
    Returns up to top_k best-matching templates, ordered by match count.

    Each matched template adds its code scaffold to the LLM context,
    guiding generation toward correct algorithmic patterns.
    """
    text = problem_statement.lower()

    scored: list[tuple[int, AlgoTemplate]] = []
    for tmpl in _TEMPLATES:
        hits = sum(1 for kw in tmpl.keywords if kw in text)
        if hits > 0:
            scored.append((hits, tmpl))

    scored.sort(key=lambda x: -x[0])
    return [t for _, t in scored[:top_k]]


def format_templates_for_prompt(templates: List[AlgoTemplate]) -> str:
    """Format retrieved templates into an LLM-ready context block."""
    if not templates:
        return ""
    parts = ["## Algorithm Templates (use as reference)\n"]
    for tmpl in templates:
        parts.append(f"### {tmpl.name}\n{tmpl.description}\n```python\n{tmpl.code}```\n")
    return "\n".join(parts)
