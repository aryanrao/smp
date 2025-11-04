import pandas as pd
from dijkstra import dijkstra
from prims import prims
from huffman import huffman_coding
from activity_selection import activity_selection
from nearest_neighbor import nearest_neighbor_tsp

def run_all_algorithms():
    """
    Runs each algorithm with its own sample data and prints the results.
    This demonstrates how each algorithm solves its *specific* problem.
    """
    
    print("="*50)
    print("🚀 EXECUTING GREEDY ALGORITHMS 🚀")
    print("="*50)
    
    # --- 1. Dijkstra's Algorithm ---
    # Graph based on Table III (A=0, B=1, C=2, D=3, E=4, F=5)
    # Using an adjacency list for the code
    graph_dijkstra = {
        'A': [('B', 10), ('C', 15), ('D', 20), ('E', 45), ('F', 40)],
        'B': [('A', 10), ('C', 5), ('D', 12), ('E', 50), ('F', 35)],
        'C': [('A', 15), ('B', 5), ('D', 7), ('E', 55), ('F', 40)],
        'D': [('A', 20), ('B', 12), ('C', 7), ('E', 60), ('F', 44)],
        'E': [('A', 45), ('B', 50), ('C', 55), ('D', 60), ('F', 10)],
        'F': [('A', 40), ('B', 35), ('C', 40), ('D', 44), ('E', 10)],
    }
    print("\n### 1. Dijkstra's Algorithm (Shortest Path) ###")
    print("Finds the shortest path from a source to all other nodes.")
    distances = dijkstra(graph_dijkstra, 'A')
    print(f"Shortest distances from node 'A':\n{distances}\n")

    # --- 2. Prim's Algorithm ---
    print("\n### 2. Prim's Algorithm (Minimum Spanning Tree) ###")
    print("Finds the minimum cost to connect all nodes in a network.")
    # We can use the same graph data
    mst_weight, mst_edges = prims(graph_dijkstra, 'A')
    print(f"Total MST Weight: {mst_weight}")
    print(f"Edges in MST: \n{mst_edges}\n")

    # --- 3. Huffman Coding ---
    print("\n### 3. Huffman Coding (Data Compression) ###")
    print("Finds an optimal prefix-free binary encoding for a string.")
    data_str = "this is a test of the greedy huffman algorithm"
    codes = huffman_coding(data_str)
    print(f"Data: '{data_str}'")
    print(f"Generated Huffman Codes:\n{codes}\n")

    # --- 4. Activity Selection ---
    print("\n### 4. Activity Selection (Interval Scheduling) ###")
    print("Finds the maximum number of non-overlapping activities.")
    # List of (start, finish) tuples
    activities = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 8), (5, 9),
                  (6, 10), (8, 11), (8, 12), (2, 13), (12, 14)]
    print(f"Input Activities: {activities}")
    selected_acts = activity_selection(activities)
    print(f"Selected Activities ({len(selected_acts)}): {selected_acts}\n")

    # --- 5. Nearest Neighbor (TSP Heuristic) ---
    print("\n### 5. Nearest Neighbor (TSP Heuristic) ###")
    print("Finds a 'good enough', but NOT optimal, path for the TSP.")
    # Using the distance matrix from Table III 
    dist_matrix = [
        # A,  B,  C,  D,  E,  F
        [ 0, 10, 15, 20, 45, 40], # A
        [10,  0,  5, 12, 50, 35], # B
        [15,  5,  0,  7, 55, 40], # C
        [20, 12,  7,  0, 60, 44], # D
        [45, 50, 55, 60,  0, 10], # E
        [40, 35, 40, 44, 10,  0]  # F
    ]
    nn_path, nn_dist = nearest_neighbor_tsp(dist_matrix, start_node=0)
    print(f"NN Path (starting at 0='A'): {nn_path}")
    print(f"NN Total Distance: {nn_dist}\n")


def qualitative_comparison():
    """
    Prints a qualitative comparison of the algorithms,
    as discussed in the research paper[cite: 1, 10, 22].
    """
    
    print("="*50)
    print("📊 QUALITATIVE COMPARISON 📊")
    print("="*50)
    print("These algorithms CANNOT be compared on speed/output for the same task.")
    print("Instead, we compare their PURPOSE and GUARANTEES:\n")
    
    data = {
        "Algorithm": [
            "Dijkstra's",
            "Prim's",
            "Huffman Coding",
            "Activity Selection",
            "Nearest Neighbor"
        ],
        "Problem Domain": [
            "Shortest Path",
            "Minimum Spanning Tree",
            "Data Compression",
            "Interval Scheduling",
            "Traveling Salesman (TSP)"
        ],
        "Optimality Guarantee": [
            "**Optimal** (non-negative weights)",
            "**Optimal**",
            "**Optimal** (for given frequencies)",
            "**Optimal**",
            "**NOT Optimal** (Heuristic)"
        ],
        "Time Complexity": [
            "O(E log V)",
            "O(E log V)",
            "O(n log n)",
            "O(n log n)",
            "O(n^2)"
        ],
        "Key Greedy Choice": [
            "Explore nearest unvisited node",
            "Add cheapest edge to growing tree",
            "Merge two lowest-frequency nodes",
            "Pick activity that *finishes* earliest",
            "Go to *absolute* nearest city"
        ]
    }
    
    # Using pandas for a clean print, but you could format this manually.
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print("\n[Data for this table was synthesized from Table I in the paper draft]")


if __name__ == "__main__":
    run_all_algorithms()
    qualitative_comparison()