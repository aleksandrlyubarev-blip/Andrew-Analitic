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

    AlgoTemplate(
        name="DFS / Backtracking",
        keywords=["dfs", "depth first", "backtrack", "backtracking", "permutation",
                  "combination", "subset", "generate all", "enumerate", "recursion"],
        description="Generic DFS/backtracking template for enumerating states.",
        code='''\
def backtrack(path: list, choices: list) -> None:
    """Enumerate all valid paths/subsets. Prune early via constraints."""
    if is_goal(path):
        results.append(path[:])
        return
    for i, choice in enumerate(choices):
        if not is_valid(path, choice):
            continue
        path.append(choice)
        backtrack(path, choices[i+1:])   # adjust slice for combinations vs permutations
        path.pop()

results: list = []
backtrack([], list(range(n)))

# Iterative DFS (stack-based, avoids recursion limit)
def dfs_iterative(graph: dict, start: int) -> list:
    visited = set()
    stack = [start]
    order = []
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for nb in reversed(graph.get(node, [])):
            if nb not in visited:
                stack.append(nb)
    return order
''',
    ),

    AlgoTemplate(
        name="Floyd-Warshall (All-Pairs Shortest Path)",
        keywords=["floyd", "all pairs", "all-pairs shortest path", "negative cycle",
                  "transitive closure", "reachability matrix"],
        description="O(V^3) all-pairs shortest path; detects negative cycles.",
        code='''\
import math

def floyd_warshall(n: int, edges: list[tuple[int,int,int]]) -> list[list[float]]:
    """
    Returns dist[i][j] = shortest path from i to j (0-indexed).
    dist[i][i] < 0 after run → negative cycle exists.
    """
    INF = math.inf
    dist = [[INF]*n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        dist[u][v] = min(dist[u][v], w)

    for k in range(n):
        for i in range(n):
            if dist[i][k] == INF:
                continue
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist
''',
    ),

    AlgoTemplate(
        name="Minimum Spanning Tree (Kruskal)",
        keywords=["minimum spanning tree", "mst", "kruskal", "minimum cost tree",
                  "spanning tree", "prim"],
        description="Kruskal MST via DSU; O(E log E).",
        code='''\
def kruskal(n: int, edges: list[tuple[int,int,int]]) -> tuple[int, list]:
    """
    Returns (total_weight, mst_edges).
    edges: [(weight, u, v), ...]  — sort by weight ascending.
    """
    edges.sort()
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y) -> bool:
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True

    total, mst = 0, []
    for w, u, v in edges:
        if union(u, v):
            total += w
            mst.append((u, v, w))
    return total, mst
''',
    ),

    AlgoTemplate(
        name="Sparse Table (RMQ)",
        keywords=["sparse table", "rmq", "range minimum query", "range maximum query",
                  "static range", "idempotent", "offline range"],
        description="O(n log n) build, O(1) query for static range min/max.",
        code='''\
import math

class SparseTable:
    """Range minimum query (change min→max or any idempotent op)."""
    def __init__(self, arr: list[int]):
        n = len(arr)
        LOG = max(1, n.bit_length())
        self.table = [[0]*n for _ in range(LOG)]
        self.table[0] = arr[:]
        for k in range(1, LOG):
            for i in range(n - (1 << k) + 1):
                self.table[k][i] = min(
                    self.table[k-1][i],
                    self.table[k-1][i + (1 << (k-1))],
                )
        self.log2 = [0] * (n + 1)
        for i in range(2, n + 1):
            self.log2[i] = self.log2[i // 2] + 1

    def query(self, l: int, r: int) -> int:
        """Minimum of arr[l..r] inclusive (0-indexed), O(1)."""
        k = self.log2[r - l + 1]
        return min(self.table[k][l], self.table[k][r - (1 << k) + 1])
''',
    ),

    AlgoTemplate(
        name="Lowest Common Ancestor (Binary Lifting)",
        keywords=["lca", "lowest common ancestor", "binary lifting", "ancestor",
                  "tree queries", "path query on tree", "kth ancestor"],
        description="O(n log n) preprocessing, O(log n) LCA queries via binary lifting.",
        code='''\
import math

class LCA:
    def __init__(self, n: int, root: int, adj: list[list[int]]):
        LOG = max(1, n.bit_length())
        self.depth = [0] * n
        self.up = [[-1]*n for _ in range(LOG)]
        # BFS to set depth and up[0]
        from collections import deque
        q = deque([root])
        visited = [False] * n
        visited[root] = True
        while q:
            u = q.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    self.depth[v] = self.depth[u] + 1
                    self.up[0][v] = u
                    q.append(v)
        self.up[0][root] = root
        for k in range(1, LOG):
            for v in range(n):
                self.up[k][v] = self.up[k-1][self.up[k-1][v]]
        self.LOG = LOG

    def lca(self, u: int, v: int) -> int:
        if self.depth[u] < self.depth[v]:
            u, v = v, u
        diff = self.depth[u] - self.depth[v]
        for k in range(self.LOG):
            if (diff >> k) & 1:
                u = self.up[k][u]
        if u == v:
            return u
        for k in range(self.LOG - 1, -1, -1):
            if self.up[k][u] != self.up[k][v]:
                u = self.up[k][u]
                v = self.up[k][v]
        return self.up[0][u]
''',
    ),

    AlgoTemplate(
        name="KMP / Z-Function",
        keywords=["kmp", "knuth morris pratt", "z-function", "z function",
                  "pattern matching", "string search", "occurrence", "period"],
        description="KMP failure function and Z-array for O(n+m) pattern matching.",
        code='''\
def kmp_search(text: str, pattern: str) -> list[int]:
    """Return all start indices of pattern in text (0-indexed)."""
    s = pattern + "#" + text
    n = len(s)
    fail = [0] * n
    j = 0
    for i in range(1, n):
        while j > 0 and s[i] != s[j]:
            j = fail[j-1]
        if s[i] == s[j]:
            j += 1
        fail[i] = j
        if j == len(pattern):
            yield i - 2 * len(pattern)    # start index in text
            j = fail[j-1]

def z_array(s: str) -> list[int]:
    """z[i] = length of longest substring starting at s[i] matching a prefix of s."""
    n = len(s)
    z = [0] * n
    z[0] = n
    l = r = 0
    for i in range(1, n):
        if i < r:
            z[i] = min(r - i, z[i - l])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] > r:
            l, r = i, i + z[i]
    return z
''',
    ),

    AlgoTemplate(
        name="Bitmask DP",
        keywords=["bitmask dp", "bitmask", "subset dp", "traveling salesman",
                  "tsp", "hamiltonian path", "assignment problem", "dp on subsets"],
        description="O(2^n × n) DP over all subsets; classic for TSP / assignment.",
        code='''\
import math

def tsp(dist: list[list[int]], n: int) -> int:
    """
    Minimum cost Hamiltonian cycle (TSP).
    dist[i][j] = cost from city i to city j.
    State: dp[mask][v] = min cost to visit exactly cities in mask, ending at v.
    """
    INF = math.inf
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0   # start at city 0

    for mask in range(1 << n):
        for u in range(n):
            if dp[mask][u] == INF:
                continue
            if not (mask >> u & 1):
                continue
            for v in range(n):
                if mask >> v & 1:
                    continue
                new_mask = mask | (1 << v)
                cost = dp[mask][u] + dist[u][v]
                if cost < dp[new_mask][v]:
                    dp[new_mask][v] = cost

    full = (1 << n) - 1
    return min(dp[full][v] + dist[v][0] for v in range(1, n))

# Enumerate all subsets of a mask (useful in subset-sum DP)
def iter_subsets(mask: int):
    sub = mask
    while sub:
        yield sub
        sub = (sub - 1) & mask
''',
    ),

    AlgoTemplate(
        name="Convex Hull",
        keywords=["convex hull", "graham scan", "andrew monotone chain",
                  "geometry", "polygon area", "point in polygon", "collinear"],
        description="Andrew's monotone chain convex hull in O(n log n).",
        code='''\
def convex_hull(points: list[tuple[int,int]]) -> list[tuple[int,int]]:
    """
    Returns vertices of convex hull in CCW order (Andrew's monotone chain).
    Collinear points on the hull are excluded (use <= for inclusive).
    """
    def cross(O, A, B):
        return (A[0]-O[0])*(B[1]-O[1]) - (A[1]-O[1])*(B[0]-O[0])

    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]

def polygon_area_2x(pts: list[tuple[int,int]]) -> int:
    """2 × signed area of polygon (positive = CCW). Divide by 2 for actual area."""
    n = len(pts)
    return abs(sum(
        pts[i][0] * pts[(i+1)%n][1] - pts[(i+1)%n][0] * pts[i][1]
        for i in range(n)
    ))
''',
    ),

    AlgoTemplate(
        name="Sprague-Grundy / Game Theory",
        keywords=["game theory", "sprague grundy", "nim", "grundy value",
                  "nimber", "impartial game", "mex", "losing position", "winning position"],
        description="Sprague-Grundy theorem: compute Grundy values (nimbers) for impartial games.",
        code='''\
from functools import lru_cache

def mex(reachable: set[int]) -> int:
    """Minimum Excludant — smallest non-negative integer not in reachable."""
    g = 0
    while g in reachable:
        g += 1
    return g

@lru_cache(maxsize=None)
def grundy(state) -> int:
    """
    Returns Grundy value of a game state.
    Implement moves(state) → iterable of next states.
    """
    return mex({grundy(s) for s in moves(state)})

def is_winning(state) -> bool:
    return grundy(state) != 0

# Multi-pile Nim: XOR all pile sizes
def nim_winner(piles: list[int]) -> bool:
    """Returns True if the current player wins Nim with given pile sizes."""
    xor = 0
    for p in piles:
        xor ^= p
    return xor != 0
''',
    ),

    AlgoTemplate(
        name="Meet in the Middle",
        keywords=["meet in the middle", "meet-in-the-middle", "split search",
                  "baby step giant step", "bsgs", "subset sum large n"],
        description="Split problem into two halves, enumerate each independently, then combine.",
        code='''\
def meet_in_middle(arr: list[int], target: int) -> int:
    """Count subsets of arr that sum to target (n up to ~40)."""
    from collections import defaultdict

    n = len(arr)
    half = n // 2
    left, right = arr[:half], arr[half:]

    def all_subset_sums(a: list[int]) -> list[int]:
        sums = [0]
        for x in a:
            sums = sums + [s + x for s in sums]
        return sums

    left_sums = sorted(all_subset_sums(left))
    right_counts: defaultdict[int, int] = defaultdict(int)
    for s in all_subset_sums(right):
        right_counts[s] += 1

    count = 0
    for s in left_sums:
        count += right_counts[target - s]
    return count
''',
    ),

    AlgoTemplate(
        name="Mo's Algorithm (Offline Range Queries)",
        keywords=["mo's algorithm", "mo algorithm", "offline queries",
                  "range queries offline", "sqrt decomposition queries", "block decomposition"],
        description="O((n+q)√n) offline range queries by sorting on √n block boundaries.",
        code='''\
import math

def mo_queries(arr: list[int], queries: list[tuple[int,int]]) -> list[int]:
    """
    Process range queries [l, r] offline.
    Implement add(i), remove(i), get_answer() for your problem.
    """
    n = len(arr)
    block = max(1, int(math.sqrt(n)))
    # Sort queries: (block of l, r ASC if block even, r DESC if block odd)
    indexed = sorted(
        enumerate(queries),
        key=lambda x: (x[1][0]//block, x[1][1] if (x[1][0]//block)%2==0 else -x[1][1])
    )

    cur_l, cur_r = 0, -1
    freq: dict[int, int] = {}
    ans_count = 0   # maintain current answer (e.g., count of distinct elements)
    answers = [0] * len(queries)

    def add(i: int):
        nonlocal ans_count
        v = arr[i]
        if freq.get(v, 0) == 0:
            ans_count += 1
        freq[v] = freq.get(v, 0) + 1

    def remove(i: int):
        nonlocal ans_count
        v = arr[i]
        freq[v] -= 1
        if freq[v] == 0:
            ans_count -= 1

    for qi, (l, r) in indexed:
        while cur_r < r: cur_r += 1; add(cur_r)
        while cur_l > l: cur_l -= 1; add(cur_l)
        while cur_r > r: remove(cur_r); cur_r -= 1
        while cur_l < l: remove(cur_l); cur_l += 1
        answers[qi] = ans_count

    return answers
''',
    ),

    AlgoTemplate(
        name="Bellman-Ford / SPFA",
        keywords=["bellman ford", "bellman-ford", "spfa", "negative weight",
                  "negative edge", "negative cycle detection", "shortest path negative"],
        description="Bellman-Ford O(VE) for negative weights; SPFA queue optimisation.",
        code='''\
def bellman_ford(n: int, edges: list[tuple[int,int,int]], src: int):
    """
    Returns (dist, has_negative_cycle).
    edges: [(u, v, weight), ...]
    """
    INF = float("inf")
    dist = [INF] * n
    dist[src] = 0

    for _ in range(n - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True
        if not updated:
            break

    # Check for negative cycles
    for u, v, w in edges:
        if dist[u] != INF and dist[u] + w < dist[v]:
            return dist, True

    return dist, False

# SPFA — Bellman-Ford with deque (faster in practice, same worst-case)
from collections import deque

def spfa(adj: list[list[tuple[int,int]]], src: int) -> list[float]:
    n = len(adj)
    dist = [float("inf")] * n
    dist[src] = 0
    in_queue = [False] * n
    q = deque([src])
    in_queue[src] = True
    while q:
        u = q.popleft()
        in_queue[u] = False
        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                if not in_queue[v]:
                    q.append(v)
                    in_queue[v] = True
    return dist
''',
    ),

    AlgoTemplate(
        name="Aho-Corasick (Multi-Pattern Matching)",
        keywords=["aho corasick", "aho-corasick", "multiple patterns",
                  "multi-pattern", "automaton", "string automaton", "dictionary matching"],
        description="O(total_pattern_len + text_len) simultaneous search of all patterns.",
        code='''\
from collections import deque

class AhoCorasick:
    def __init__(self, patterns: list[str]):
        self.goto: list[dict[str,int]] = [{}]
        self.fail = [0]
        self.output: list[list[int]] = [[]]   # pattern indices ending at each node

        # Build goto
        for pi, pat in enumerate(patterns):
            cur = 0
            for ch in pat:
                if ch not in self.goto[cur]:
                    self.goto[cur][ch] = len(self.goto)
                    self.goto.append({})
                    self.fail.append(0)
                    self.output.append([])
                cur = self.goto[cur][ch]
            self.output[cur].append(pi)

        # Build fail links (BFS)
        q = deque()
        for ch, nxt in self.goto[0].items():
            self.fail[nxt] = 0
            q.append(nxt)
        while q:
            u = q.popleft()
            for ch, v in self.goto[u].items():
                f = self.fail[u]
                while f and ch not in self.goto[f]:
                    f = self.fail[f]
                self.fail[v] = self.goto[f].get(ch, 0)
                if self.fail[v] == v:
                    self.fail[v] = 0
                self.output[v] += self.output[self.fail[v]]
                q.append(v)

    def search(self, text: str) -> list[tuple[int,int]]:
        """Returns [(end_pos, pattern_idx), ...] for all matches."""
        cur = 0
        results = []
        for i, ch in enumerate(text):
            while cur and ch not in self.goto[cur]:
                cur = self.fail[cur]
            cur = self.goto[cur].get(ch, 0)
            for pi in self.output[cur]:
                results.append((i, pi))
        return results
''',
    ),

    AlgoTemplate(
        name="Heavy-Light Decomposition",
        keywords=["heavy light decomposition", "hld", "heavy path",
                  "path update tree", "path query tree", "tree path"],
        description="HLD: reduce tree path queries to O(log^2 n) range queries on an array.",
        code='''\
class HLD:
    """
    Heavy-Light Decomposition for path queries on trees.
    Combine with a Segment Tree or Fenwick Tree for the actual queries.
    """
    def __init__(self, n: int, root: int, adj: list[list[int]]):
        self.n = n
        self.parent = [-1] * n
        self.depth  = [0]  * n
        self.size   = [1]  * n
        self.heavy  = [-1] * n   # heavy child
        self.head   = [0]  * n   # chain head
        self.pos    = [0]  * n   # position in flat array

        self._dfs_size(root, adj)
        self._dfs_hld(root, root, 0, adj)

    def _dfs_size(self, root: int, adj):
        stack = [(root, -1, False)]
        while stack:
            u, p, returning = stack.pop()
            if returning:
                for v in adj[u]:
                    if v == p: continue
                    self.size[u] += self.size[v]
                    if self.heavy[u] == -1 or self.size[v] > self.size[self.heavy[u]]:
                        self.heavy[u] = v
            else:
                self.parent[u] = p
                stack.append((u, p, True))
                for v in adj[u]:
                    if v != p:
                        self.depth[v] = self.depth[u] + 1
                        stack.append((v, u, False))

    def _dfs_hld(self, u: int, h: int, cur_pos: list, adj):
        timer = [0]
        stack = [(u, h)]
        while stack:
            node, head = stack.pop()
            self.head[node] = head
            self.pos[node]  = timer[0]; timer[0] += 1
            if self.heavy[node] != -1:
                stack.append((self.heavy[node], head))
            for v in adj[node]:
                if v != self.parent[node] and v != self.heavy[node]:
                    stack.append((v, v))

    def path_query(self, u: int, v: int, query_fn) -> int:
        """
        Aggregate query_fn(l, r) over all chain segments on path u→v.
        query_fn(l, r) should return the answer for the flat array range [l, r].
        """
        result = 0   # adjust identity for your monoid (0 for sum, inf for min)
        while self.head[u] != self.head[v]:
            if self.depth[self.head[u]] < self.depth[self.head[v]]:
                u, v = v, u
            result += query_fn(self.pos[self.head[u]], self.pos[u])
            u = self.parent[self.head[u]]
        if self.depth[u] > self.depth[v]:
            u, v = v, u
        result += query_fn(self.pos[u], self.pos[v])
        return result
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
