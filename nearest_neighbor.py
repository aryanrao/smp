def nearest_neighbor_tsp(dist_matrix, start_node=0):
    """
    Finds an approximate solution to the TSP using the
    Nearest Neighbor greedy heuristic.
    This is NOT optimal. 

    Args:
        dist_matrix (list of list): A 2D list representing the
                                    distances between cities.
                                    dist_matrix[i][j] is dist from i to j.
        start_node (int): The index of the starting city.

    Returns:
        (list, int): A tuple containing:
                     1. The path (list of city indices).
                     2. The total distance of the path.
    """
    num_cities = len(dist_matrix)
    if num_cities == 0:
        return [], 0

    visited = [False] * num_cities
    path = [start_node]
    total_distance = 0
    current_city = start_node
    visited[current_city] = True

    for _ in range(num_cities - 1):
        nearest_city = -1
        min_distance = float('inf')

        # Find the nearest unvisited city
        for next_city in range(num_cities):
            if not visited[next_city] and dist_matrix[current_city][next_city] < min_distance:
                min_distance = dist_matrix[current_city][next_city]
                nearest_city = next_city
        
        if nearest_city == -1:
            # Should not happen in a connected graph
            break

        # Move to the nearest city
        current_city = nearest_city
        visited[current_city] = True
        path.append(current_city)
        total_distance += min_distance

    # Add the distance to return to the start
    total_distance += dist_matrix[path[-1]][start_node]
    path.append(start_node)

    return path, total_distance