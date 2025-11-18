import time
import random
import pandas as pd
import string
import sys

# Import our implemented algorithms
from dijkstra import dijkstra
from prims import prims
from huffman import huffman_coding
from activity_selection import activity_selection
from nearest_neighbor import nearest_neighbor_tsp

# --- DATA GENERATOR FUNCTIONS (Unchanged) ---

def generate_graph(n_nodes, n_edges):
    """
    Generates a random, connected, undirected graph as an adjacency list.
    """
    if n_edges < n_nodes - 1:
        n_edges = n_nodes - 1
    
    graph = {i: [] for i in range(n_nodes)}
    edges = set()

    # 1. Ensure connectivity
    for i in range(n_nodes - 1):
        weight = random.randint(1, 100)
        u, v = i, i + 1
        graph[u].append((v, weight))
        graph[v].append((u, weight))
        edges.add(tuple(sorted((u, v))))

    # 2. Add remaining edges
    target_edges = n_edges
    while len(edges) < target_edges and len(edges) < (n_nodes * (n_nodes - 1) / 2):
        u, v = random.sample(range(n_nodes), 2)
        if u == v:
            continue
        
        edge = tuple(sorted((u, v)))
        if edge not in edges:
            weight = random.randint(1, 100)
            graph[u].append((v, weight))
            graph[v].append((u, weight))
            edges.add(edge)
            
    str_graph = {f'N{k}': [(f'N{n}', w) for n, w in v] for k, v in graph.items()}
    start_node = 'N0'
    return (str_graph, start_node)

def generate_dist_matrix(n_cities):
    """Generates a symmetric distance matrix for the TSP."""
    matrix = [[0] * n_cities for _ in range(n_cities)]
    for i in range(n_cities):
        for j in range(i + 1, n_cities):
            dist = random.randint(1, 1000)
            matrix[i][j] = dist
            matrix[j][i] = dist
    return (matrix, 0)

def generate_activities(n_activities):
    """Generates a list of (start, finish) activity tuples."""
    activities = []
    for _ in range(n_activities):
        start = random.randint(0, 1000)
        duration = random.randint(1, 100)
        finish = start + duration
        activities.append((start, finish))
    return (activities,)

def generate_string(length):
    """Generates a random string for Huffman coding."""
    chars = string.ascii_lowercase + " " * 10
    s = "".join(random.choices(chars, k=length))
    return (s,)

# --- FAST DEMONSTRATION FUNCTION ---

def time_algo_run(algo_name, func_to_run, data_gen_func, data_gen_args):
    """
    Runs an algorithm ONCE and times it with perf_counter for speed.
    This is a "demo," not a formal "benchmark."
    """
    print(f"  Timing {algo_name} (N={data_gen_args[0]})...", end='', flush=True)
    
    # 1. Generate the data
    data = data_gen_func(*data_gen_args)
    
    # 2. Run and Time (ONCE)
    # This is much faster than timeit + tracemalloc
    start_time = time.perf_counter()
    try:
        func_to_run(*data)
    except Exception as e:
        print(f"Error: {e}")
        return {}
    end_time = time.perf_counter()
    
    time_ms = (end_time - start_time) * 1000
    
    print(" Done.")
    
    return {
        "Algorithm": algo_name,
        "Input Size (N)": data_gen_args[0],
        "Time (ms)": time_ms
    }

# --- MAIN EXECUTION ---

def run_fast_demo():
    """
    Runs demonstrations for all algorithms at different scales.
    """
    print("="*60)
    print("🚀 RUNNING GREEDY ALGORITHM FAST DEMO 🚀")
    print("This script runs each algorithm ONCE to show performance.")
    print("="*60)
    
    results = []
    
    # --- Define Scales (Same as before, will run fast now) ---
    GRAPH_SCALES = [
        {"N": 50, "E": 200},
        {"N": 200, "E": 1000},
        {"N": 500, "E": 5000}
    ]
    LIST_SCALES = [500, 5000, 20000]
    STRING_SCALES = [5000, 50000, 200000]

    # --- 1. Dijkstra's Benchmark (O(E log V)) ---
    print("\n[Timing Graph Algorithms...]")
    for scale in GRAPH_SCALES:
        n, e = scale["N"], scale["E"]
        results.append(time_algo_run("Dijkstra's", dijkstra, 
                                      generate_graph, (n, e)))

    # --- 2. Prim's Benchmark (O(E log V)) ---
    for scale in GRAPH_SCALES:
        n, e = scale["N"], scale["E"]
        results.append(time_algo_run("Prim's", prims, 
                                      generate_graph, (n, e)))

    # --- 3. Nearest Neighbor Benchmark (O(N^2)) ---
    for scale in GRAPH_SCALES:
        n = scale["N"]
        results.append(time_algo_run("Nearest Neighbor", nearest_neighbor_tsp, 
                                      generate_dist_matrix, (n,)))

    # --- 4. Huffman Coding Benchmark (O(N log N)) ---
    print("\n[Timing String/List Algorithms...]")
    for n in STRING_SCALES:
        results.append(time_algo_run("Huffman Coding", huffman_coding, 
                                      generate_string, (n,)))

    # --- 5. Activity Selection Benchmark (O(N log N)) ---
    for n in LIST_SCALES:
        results.append(time_algo_run("Activity Selection", activity_selection, 
                                      generate_activities, (n,)))

    print("\n✅ Demo complete.")
    
    # --- Display Results ---
    print("\n" + "="*60)
    print("DEMONSTRATION RESULTS (TIME) ")
    print("="*60)
    
    df = pd.DataFrame(results)
    
    # Format for better readability
    df["Time (ms)"] = df["Time (ms)"].apply(lambda x: f"{x:.4f}")
    
    # Group by Algorithm for cleaner table
    for algo_name, group in df.groupby("Algorithm"):
        print(f"\n--- {algo_name} ---")
        group = group.reset_index(drop=True)
        print(group[['Input Size (N)', 'Time (ms)']].to_string(index=False))

    
    print("\n\n" + "="*60)
    print("--- Analysis ---")
    print("This fast demo shows the relative time for a SINGLE RUN.")
    print("You can see the O(N^2) 'Nearest Neighbor' and O(N log N) 'Huffman' scale up.")
    print("'Activity Selection' is extremely fast as it's just sorting.")


if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        print("Error: This script requires the 'pandas' library to display results.")
        print("Please run: pip install pandas")
        sys.exit(1)
        
    # Set higher recursion depth for Huffman coding on large strings
    sys.setrecursionlimit(2000) 
    
    run_fast_demo()