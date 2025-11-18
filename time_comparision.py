# compare_greedy_algorithms.py
# Multi-algorithm benchmarking script
# See top of file for requirements and usage.

import random
import time
import math
from heapq import heappush, heappop
import matplotlib.pyplot as plt
import numpy as np

def timeit(func):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        res = func(*args, **kwargs)
        t1 = time.perf_counter()
        return res, (t1 - t0)
    return wrapper

@timeit
def dijkstra(n, edges, source=0):
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    dist = [math.inf] * n
    dist[source] = 0.0
    heap = [(0.0, source)]
    visited = [False] * n
    while heap:
        d, u = heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heappush(heap, (nd, v))
    return dist

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[ry] < self.rank[rx]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1
        return True

@timeit
def kruskal(n, edges):
    uf = UnionFind(n)
    mst = []
    edges_sorted = sorted(edges, key=lambda e: e[2])
    for u, v, w in edges_sorted:
        if uf.union(u, v):
            mst.append((u, v, w))
            if len(mst) == n - 1:
                break
    return mst

@timeit
def nearest_neighbor_tsp(coords):
    n = len(coords)
    if n == 0:
        return [], 0.0
    def dist(i, j):
        (x1, y1), (x2, y2) = coords[i], coords[j]
        return math.hypot(x1 - x2, y1 - y2)
    visited = [False] * n
    tour = [0]
    visited[0] = True
    total = 0.0
    cur = 0
    for _ in range(n - 1):
        best = None
        bestd = float('inf')
        for j in range(n):
            if not visited[j]:
                d = dist(cur, j)
                if d < bestd:
                    bestd = d
                    best = j
        tour.append(best)
        visited[best] = True
        total += bestd
        cur = best
    total += dist(cur, 0)
    tour.append(0)
    return tour, total

@timeit
def greedy_insertion_tsp(coords):
    n = len(coords)
    if n == 0:
        return [], 0.0
    def dist(i, j):
        (x1, y1), (x2, y2) = coords[i], coords[j]
        return math.hypot(x1 - x2, y1 - y2)
    if n == 1:
        return [0, 0], 0.0
    tour = [0, 1, 0]
    remaining = set(range(2, n))
    while remaining:
        best_node = None
        best_increase = float('inf')
        best_pos = None
        for node in list(remaining):
            for i in range(len(tour) - 1):
                a = tour[i]
                b = tour[i + 1]
                increase = dist(a, node) + dist(node, b) - dist(a, b)
                if increase < best_increase:
                    best_increase = increase
                    best_node = node
                    best_pos = i + 1
        tour.insert(best_pos, best_node)
        remaining.remove(best_node)
    total = 0.0
    for i in range(len(tour) - 1):
        total += dist(tour[i], tour[i + 1])
    return tour, total

@timeit
def fractional_knapsack(items, capacity):
    items_sorted = sorted(items, key=lambda x: x[0] / x[1], reverse=True)
    total_value = 0.0
    taken = []
    for v, w in items_sorted:
        if capacity <= 0:
            break
        if w <= capacity:
            total_value += v
            capacity -= w
            taken.append((v, w, 1.0))
        else:
            frac = capacity / w
            total_value += v * frac
            taken.append((v, w, frac))
            capacity = 0
            break
    return total_value, taken

@timeit
def activity_selection(activities):
    activities_sorted = sorted(activities, key=lambda x: x[1])
    selected = []
    last_finish = -1e18
    for s, f in activities_sorted:
        if s >= last_finish:
            selected.append((s, f))
            last_finish = f
    return selected

def generate_random_graph(n, density=0.1, weight_range=(1, 100), seed=None):
    if seed is not None:
        random.seed(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < density:
                w = random.randint(*weight_range)
                edges.append((u, v, w))
    if len(edges) < n - 1:
        for i in range(n - 1):
            edges.append((i, i + 1, random.randint(*weight_range)))
    return edges

def generate_random_points(n, coord_range=(0, 1000), seed=None):
    if seed is not None:
        random.seed(seed)
    return [(random.uniform(*coord_range), random.uniform(*coord_range)) for _ in range(n)]

def generate_items(m, value_range=(10, 500), weight_range=(1, 100), seed=None):
    if seed is not None:
        random.seed(seed)
    return [(random.randint(*value_range), random.randint(*weight_range)) for _ in range(m)]

def generate_activities(m, time_range=(0, 1000), seed=None):
    if seed is not None:
        random.seed(seed)
    activities = []
    for _ in range(m):
        a = random.randint(*time_range)
        b = random.randint(a, time_range[1])
        activities.append((a, b))
    return activities

def run_benchmarks(run_demo=True):
    graph_sizes = [30, 60, 100]
    tsp_sizes = [20, 40, 60]
    knapsack_sizes = [100, 200, 400]
    activity_sizes = [100, 200, 400]
    repeats = 3

    results = {
        "dijkstra": [],
        "kruskal": [],
        "tsp_nn": [],
        "tsp_gi": [],
        "knapsack": [],
        "activity": []
    }

    for n in graph_sizes:
        td = 0.0
        tk = 0.0
        for r in range(repeats):
            edges = generate_random_graph(n, density=min(0.08 + 0.01*(n/50), 0.3), seed=r + n)
            _, t1 = dijkstra(n, edges, source=0)
            _, t2 = kruskal(n, edges)
            td += t1
            tk += t2
        results["dijkstra"].append(td / repeats)
        results["kruskal"].append(tk / repeats)

    for n in tsp_sizes:
        tnn = 0.0
        tgi = 0.0
        for r in range(repeats):
            pts = generate_random_points(n, seed=r + n)
            _, t1 = nearest_neighbor_tsp(pts)
            _, t2 = greedy_insertion_tsp(pts)
            tnn += t1
            tgi += t2
        results["tsp_nn"].append(tnn / repeats)
        results["tsp_gi"].append(tgi / repeats)

    for m in knapsack_sizes:
        tk = 0.0
        for r in range(repeats):
            items = generate_items(m, seed=r + m)
            capacity = int(sum(w for _, w in items) * 0.25)
            _, t = fractional_knapsack(items, capacity)
            tk += t
        results["knapsack"].append(tk / repeats)

    for m in activity_sizes:
        ta = 0.0
        for r in range(repeats):
            acts = generate_activities(m, seed=r + m)
            _, t = activity_selection(acts)
            ta += t
        results["activity"].append(ta / repeats)

    print("Benchmark results (average times in seconds):")
    print("Graph sizes:", graph_sizes)
    print("Dijkstra:", np.array(results["dijkstra"]))
    print("Kruskal:", np.array(results["kruskal"]))
    print("TSP sizes:", tsp_sizes)
    print("Nearest Neighbor:", np.array(results["tsp_nn"]))
    print("Greedy Insertion:", np.array(results["tsp_gi"]))
    print("Knapsack sizes:", knapsack_sizes)
    print("Knapsack times:", np.array(results["knapsack"]))
    print("Activity sizes:", activity_sizes)
    print("Activity times:", np.array(results["activity"]))

    if run_demo:
        plt.figure()
        plt.plot(graph_sizes, results["dijkstra"], marker='o', label="Dijkstra")
        plt.plot(graph_sizes, results["kruskal"], marker='o', label="Kruskal")
        plt.xlabel("Number of nodes")
        plt.ylabel("Time (s)")
        plt.title("Graph Algorithms: Execution Time vs Number of Nodes")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(tsp_sizes, results["tsp_nn"], marker='o', label="Nearest Neighbor")
        plt.plot(tsp_sizes, results["tsp_gi"], marker='o', label="Greedy Insertion")
        plt.xlabel("Number of points")
        plt.ylabel("Time (s)")
        plt.title("TSP Heuristics: Execution Time vs Number of Points")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(knapsack_sizes, results["knapsack"], marker='o')
        plt.xlabel("Number of items")
        plt.ylabel("Time (s)")
        plt.title("Fractional Knapsack: Execution Time vs Number of Items")
        plt.show()

        plt.figure()
        plt.plot(activity_sizes, results["activity"], marker='o')
        plt.xlabel("Number of activities")
        plt.ylabel("Time (s)")
        plt.title("Activity Selection: Execution Time vs Number of Activities")
        plt.show()

    return results

if __name__ == "__main__":
    res = run_benchmarks(run_demo=True)
    for k, v in res.items():
        print(f"{k}: {v}")
