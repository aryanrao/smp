import heapq

def prims(graph, start_node):
    """
    Implements Prim's algorithm to find the Minimum Spanning Tree (MST) of a graph.
    [cite: 102, 318, 381-384]

    Args:
        graph (dict): An adjacency list representation of the UNDIRECTED graph.
                      Format: {node: [(neighbor, weight), ...]}
        start_node: The node to start building the MST from.

    Returns:
        (int, set): A tuple containing:
                    1. The total weight of the MST.
                    2. A set of edges (tuples) in the MST, e.g., {('A', 'B', 5), ...}
    """
    mst_edges = set()
    total_weight = 0
    visited = set()
    
    # Priority queue stores tuples of (weight, from_node, to_node)
    pq = [(0, start_node, start_node)] # (weight, from, to)

    while pq and len(visited) < len(graph):
        weight, from_node, to_node = heapq.heappop(pq)

        if to_node in visited:
            continue

        # Add the new node and edge to the MST
        visited.add(to_node)
        total_weight += weight
        if from_node != to_node: # Avoid adding the initial (0, start, start) edge
            # Store edges in a consistent order (e.g., alphabetical)
            edge = tuple(sorted((from_node, to_node))) + (weight,)
            mst_edges.add(edge)

        # Add all edges from the newly visited node to the priority queue
        for neighbor, edge_weight in graph[to_node]:
            if neighbor not in visited:
                heapq.heappush(pq, (edge_weight, to_node, neighbor))

    return total_weight, mst_edges