import math
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