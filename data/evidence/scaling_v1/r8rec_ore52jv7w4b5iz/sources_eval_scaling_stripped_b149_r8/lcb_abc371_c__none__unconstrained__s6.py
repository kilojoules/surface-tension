import sys
from itertools import permutations

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    ptr = 0
    N = int(input_data[ptr])
    ptr += 1
    
    # Graph G edges
    MG = int(input_data[ptr])
    ptr += 1
    G_edges_list = []
    for _ in range(MG):
        u = int(input_data[ptr])
        v = int(input_data[ptr+1])
        G_edges_list.append((u, v))
        ptr += 2
        
    # Graph H edges
    MH = int(input_data[ptr])
    ptr += 1
    H_edges_list = []
    for _ in range(MH):
        u = int(input_data[ptr])
        v = int(input_data[ptr+1])
        H_edges_list.append((u, v))
        ptr += 2
        
    # Cost matrix A
    # The costs are provided in a flattened triangular format:
    # A[1,2], A[1,3]... A[1,N], A[2,3]... A[2,N]...
    # We map these to a dictionary for easy access.
    costs_flat = list(map(int, input_data[ptr:]))
    
    # Pre-calculate the indices for the cost matrix to avoid loops
    # cost_map[(i, j)] = cost to toggle edge between i and j (i < j)
    # We generate the pairs (i, j) in the exact order they appear in the input.
    all_pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    cost_map = {pair: cost for pair, cost in zip(all_pairs, costs_flat)}

    # Adjacency matrices for G and H (using sets of frozen sets for fast lookup)
    # Vertices are 1-indexed
    G_adj = {frozenset([u, v]) for u, v in G_edges_list}
    H_adj = {frozenset([u, v]) for u, v in H_edges_list}

    # We need to find a permutation P of (1...N) that minimizes:
    # Sum_{i < j} cost(P_i, P_j) if (edge(i, j) in G != edge(P_i, P_j) in H)
    
    # To optimize, we pre-calculate the cost for every pair (i, j) 
    # based on whether G has an edge and H has an edge.
    # However, the cost depends on the permutation P.
    # Let's define a helper to calculate total cost for a permutation.
    
    # We can't use a loop, so we use a generator expression inside sum().
    # The cost is incurred if:
    # (edge (i, j) exists in G) XOR (edge (P[i], P[j]) exists in H)
    
    # Since N is small (up to 8), N! is 40,320.
    # We iterate through all permutations of (1...N).
    
    # To avoid loops in the cost calculation, we use the pre-defined all_pairs.
    # But the cost A_{i,j} is associated with the vertices in H.
    # If we map vertex i in G to P[i] in H, the cost to change the edge 
    # between P[i] and P[j] is A_{min(P[i], P[j]), max(P[i], P[j])}.
    
    # Let's refine:
    # For a permutation P, the total cost is:
    # sum(cost_map[(min(P[i], P[j]), max(P[i], P[j]))] 
    #     for i, j in all_pairs 
    #     if (frozenset([i, j]) in G_adj) != (frozenset([P[i-1], P[j-1]]) in H_adj))
    # Wait, the indices in all_pairs are 1-based. Let's adjust.
    
    # Correct logic:
    # For every pair of vertices (i, j) in G, we check if their connectivity 
    # matches the connectivity of their mapped vertices (P[i], P[j]) in H.
    # If they differ, we pay the cost to toggle the edge (P[i], P[j]) in H.
    
    # Using a list comprehension to evaluate all permutations.
    # We use a tuple for P to allow indexing.
    
    ans = min(
        sum(
            cost_map[(min(p[i-1], p[j-1]), max(p[i-1], p[j-1]))]
            for i, j in all_pairs
            if (frozenset([i, j]) in G_adj) != (frozenset([p[i-1], p[j-1]]) in H_adj)
        )
        for p in permutations(range(1, N + 1))
    )
    
    print(ans)

if __name__ == "__main__":
    solve()