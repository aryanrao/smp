Comparative Analysis of Greedy Algorithms
This repository contains Python implementations of several foundational greedy algorithms, created to support the research paper: "Comparative Analysis of Greedy Algorithms Across Diverse Industries: Evaluating Trade-offs in Real-World Applications."

The primary goal of this code is to provide a practical demonstration of how different greedy strategies work and to highlight the critical difference between optimal greedy algorithms and non-optimal greedy heuristics.

📜 Project Overview
This project is not a single, unified application. It is a collection of distinct algorithms, each solving a different, classic computer science problem. The main_comparison.py script runs all of them sequentially to demonstrate their individual outputs and then prints a qualitative comparison that summarizes the core findings of our research.

The key takeaway is that "greedy" is not a single type of algorithm but a design paradigm.


Optimal Algorithms: For some problems (like MST or Activity Selection), the greedy choice provably leads to the global optimum.




Heuristics: For other, NP-hard problems (like TSP), the greedy choice is a "best guess" that is fast but not guaranteed to be optimal.

⚙️ Algorithms Included
This repository implements the following 5 algorithms from scratch:


dijkstra.py: Implements Dijkstra's Algorithm to find the shortest path in a graph with non-negative edge weights (Optimal) .


prims.py: Implements Prim's Algorithm to find a Minimum Spanning Tree (MST) in an undirected, weighted graph (Optimal) .


huffman.py: Implements Huffman Coding to generate an optimal, prefix-free binary code for a given set of character frequencies (Optimal) .

activity_selection.py: Implements the optimal greedy solution for the Activity Selection (Interval Scheduling) problem by sorting by finish times.


nearest_neighbor.py: Implements the Nearest Neighbor heuristic for the Traveling Salesman Problem (TSP) (Non-Optimal) .

🚀 How to Run
No external libraries are required, with the exception of pandas, which is used only for formatting the final comparison table.

Clone the repository:

Bash

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
(Optional) Install pandas for table formatting:

Bash

pip install pandas

Run the main comparison script:

Bash

python main_comparison.py

📊 Example Output:
Running the main_comparison.py script will first execute each algorithm on its own sample data and print the results. It will then conclude with a qualitative comparison table, as seen in the research paper:

==================================================
📊 QUALITATIVE COMPARISON 📊
==================================================
These algorithms CANNOT be compared on speed/output for the same task.
Instead, we compare their PURPOSE and GUARANTEES:

           Algorithm         Problem Domain Optimality Guarantee Time Complexity        Key Greedy Choice
          Dijkstra's          Shortest Path    **Optimal** (non-negative weights)       O(E log V)   Explore nearest unvisited node
              Prim's  Minimum Spanning Tree                              **Optimal** O(E log V)   Add cheapest edge to growing tree
      Huffman Coding       Data Compression   **Optimal** (for given frequencies)     O(n log n)   Merge two lowest-frequency nodes
  Activity Selection    Interval Scheduling                              **Optimal** O(n log n)   Pick activity that *finishes* earliest
    Nearest Neighbor  Traveling Salesman (TSP)                **NOT Optimal** (Heuristic)          O(n^2)   Go to *absolute* nearest city
Disclaimer
This code is intended for educational and illustrative purposes to support a university project. The sample data is hard-coded, and the implementations are designed for clarity rather than production-level performance.
