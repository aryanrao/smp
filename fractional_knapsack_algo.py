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