import heapq
from collections import Counter

class HuffmanNode:
    """A node for the Huffman tree."""
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    # This is needed to make the node comparable in the priority queue
    def __lt__(self, other):
        return self.freq < other.freq

def huffman_coding(data_string):
    """
    Implements Huffman Coding for a given string.
    [cite: 102, 318, 385-388]

    Args:
        data_string (str): The input string to compress.

    Returns:
        dict: A dictionary mapping each character to its binary Huffman code.
    """
    if not data_string:
        return {}

    # 1. Calculate frequencies
    frequencies = Counter(data_string)

    # 2. Build the priority queue
    pq = [HuffmanNode(char, freq) for char, freq in frequencies.items()]
    heapq.heapify(pq)

    # 3. Build the Huffman Tree
    while len(pq) > 1:
        # Pop the two nodes with the lowest frequencies
        left = heapq.heappop(pq)
        right = heapq.heappop(pq)

        # Create a new internal node
        # Its character is None, and its frequency is the sum
        merged_freq = left.freq + right.freq
        merged_node = HuffmanNode(None, merged_freq)
        merged_node.left = left
        merged_node.right = right

        # Push the new node back into the queue
        heapq.heappush(pq, merged_node)

    # 4. Generate codes by traversing the tree
    # The queue now has only one node, the root
    root = pq[0]
    codes = {}
    
    def generate_codes(node, current_code):
        if node is None:
            return
        
        # If it's a leaf node, we have a code
        if node.char is not None:
            codes[node.char] = current_code
            return

        generate_codes(node.left, current_code + "0")
        generate_codes(node.right, current_code + "1")

    # Start traversal from the root with an empty code
    generate_codes(root, "")
    
    return codes