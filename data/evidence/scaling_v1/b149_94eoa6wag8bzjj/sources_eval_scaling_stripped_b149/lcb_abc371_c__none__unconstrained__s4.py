import sys
from itertools import permutations

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    ptr = 0
    N = int(input_data[ptr])
    ptr += 1
    
    # Graph G edges
    MG = int(input_data[ptr])
    ptr += 1
    G_edges = []
    for _ in range(MG):
        G_edges.append((int(input_data[ptr]), int(input_data[ptr+1])))
        ptr += 2
        
    # Graph H edges
    MH = int(input_data[ptr])
    ptr += 1
    H_edges = []
    for _ in range(MH):
        H_edges.append((int(input_data[ptr]), int(input_data[ptr+1])))
        ptr += 2
        
    # Cost matrix A
    # A is provided as a flattened upper triangle
    # A[i][j] where i < j
    # We'll store it in a dictionary for easy access: (i, j) -> cost
    # The input order is A_{1,2}, A_{1,3}... A_{1,N}, A_{2,3}... A_{N-1,N}
    costs_flat = input_data[ptr:]
    
    # Build the cost mapping
    # We generate the pairs (i, j) in the exact order they appear in the input
    all_pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    cost_map = {pair: int(cost) for pair, cost in zip(all_pairs, costs_flat)}

    # Adjacency matrices for G and H
    # Using sets of frozensets for edge lookups
    adj_G = {frozenset(edge) for edge in G_edges}
    adj_H = {frozenset(edge) for edge in H_edges}

    # We need to find a permutation P of (1...N) that minimizes:
    # Sum_{i < j} cost(P_i, P_j) if (edge(i, j) in G != edge(P_i, P_j) in H)
    
    # Pre-calculate all possible pairs (i, j) with i < j
    pairs_i_j = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    
    # The cost function for a specific permutation P
    # P is a tuple where P[i-1] is the image of vertex i
    # Note: the problem says edge (i, j) in G <=> edge (P_i, P_j) in H
    # Let's use 0-indexing for the permutation tuple P
    # Vertex i (1-indexed) maps to P[i-1]
    
    def calculate_cost(P):
        # P is a permutation of (1, ..., N)
        # For every pair (i, j) with 1 <= i < j <= N:
        # Check if edge (i, j) exists in G and if edge (P[i-1], P[j-1]) exists in H
        # If they differ, add cost A_{P[i-1], P[j-1]}
        # Note: A_{u, v} is defined for u < v, so we must sort (P[i-1], P[j-1])
        
        total = 0
        for i, j in pairs_i_j:
            has_G = frozenset((i, j)) in adj_G
            # The corresponding edge in H is between the mapped vertices
            u, v = P[i-1], P[j-1]
            has_H = frozenset((u, v)) in adj_H
            
            if has_G != has_H:
                # Cost is A_{min(u,v), max(u,v)}
                total += cost_map[(u, v) if u < v else (v, u)]
        return total

    # Try all permutations of (1, ..., N)
    # N <= 8, so N! <= 40320, which is feasible in Python
    all_perms = permutations(range(1, N + 1))
    ans = min(calculate_cost(P) for P in all_perms)
    
    print(ans)

if __name__ == "__main__":
    solve()