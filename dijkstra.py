import heapq

def dijkstra(graph, start_node):
    """
    Implements Dijkstra's algorithm to find the shortest path from a start node
    to all other nodes in a graph with non-negative edge weights.
    [cite: 102, 318, 378-380]

    Args:
        graph (dict): An adjacency list representation of the graph.
                      Format: {node: [(neighbor, weight), ...]}
        start_node: The node to start the search from.

    Returns:
        dict: A dictionary mapping each node to its shortest distance from the start_node.
    """
    # Priority queue stores tuples of (distance, node)
    pq = [(0, start_node)]
    # Dictionary to store the shortest distances found so far
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0

    while pq:
        # Get the node with the smallest distance
        current_distance, current_node = heapq.heappop(pq)

        # If we've already found a shorter path, skip
        if current_distance > distances[current_node]:
            continue

        # Explore neighbors
        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight

            # If a new, shorter path is found
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances