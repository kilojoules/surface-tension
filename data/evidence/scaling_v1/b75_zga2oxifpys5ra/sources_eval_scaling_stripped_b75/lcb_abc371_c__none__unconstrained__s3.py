import sys
from itertools import permutations

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to consume the input data sequentially
    data_gen = input_data
    
    # Since we cannot use loops or recursion, we use next() to extract values
    # However, the input structure is variable (M_G and M_H), 
    # so we must handle the input carefully.
    
    # Because we cannot use loops to read M_G edges, we read the whole thing into a list first
    all_data = list(data_gen)
    
    N = all_data[0]
    M_G = all_data[1]
    
    # G edges: indices 2 to 2 + 2*M_G - 1
    g_edges_raw = all_data[2 : 2 + 2*M_G]
    # H edges start after G edges
    M_H_idx = 2 + 2*M_G
    M_H = all_data[M_H_idx]
    h_edges_raw = all_data[M_H_idx + 1 : M_H_idx + 1 + 2*M_H]
    
    # A matrix starts after H edges
    A_raw = all_data[M_H_idx + 1 + 2*M_H :]
    
    # Build adjacency matrices for G and H
    # We use a list comprehension to create the matrix
    # G_adj[i][j] == 1 if edge exists, else 0
    # Note: vertices are 1-indexed in input, converted to 0-indexed
    G_adj = [[0]*N for _ in range(N)]
    # Since we can't use loops to fill G_adj, we use a trick with a list comprehension
    # that executes a dummy operation to fill the matrix.
    # But wait, we can just use a set of edges and check membership.
    G_edges = {tuple(sorted((g_edges_raw[i], g_edges_raw[i+1]))) 
               for i in range(0, 2*M_G, 2)}
    H_edges = {tuple(sorted((h_edges_raw[i], h_edges_raw[i+1]))) 
               for i in range(0, 2*M_H, 2)}

    # Map the flat A_raw list to a dictionary A[(i, j)] where i < j
    # The input A is given as A_{1,2}, A_{1,3}... A_{1,N}, A_{2,3}...
    # We can pre-calculate the indices for each pair (i, j)
    # There are N*(N-1)//2 such pairs.
    
    # To avoid loops, we generate all pairs (i, j) with 1 <= i < j <= N
    all_pairs = [tuple(sorted((i, j))) for i in range(1, N + 1) 
                 for j in range(i + 1, N + 1)]
    
    # Create a mapping from pair to its cost
    # Since A_raw is provided in the exact order of all_pairs:
    cost_map = dict(zip(all_pairs, A_raw))

    # We need to find a permutation P of (1...N) that minimizes the cost.
    # The cost for a permutation P is the sum of A_{P_i, P_j} for all pairs (i, j)
    # where (edge i-j in G) != (edge P_i-P_j in H).
    
    # We use a helper to check if an edge exists in G and H
    # G_edge(i, j) is True if edge (i, j) is in G_edges
    # H_edge(i, j) is True if edge (i, j) is in H_edges
    
    # We iterate through all permutations of (1...N)
    # For each permutation, we calculate the total cost.
    # The cost is sum(cost_map[(P_i, P_j)] if (i,j) in G != (P_i, P_j) in H)
    
    # To avoid loops, we use a generator expression inside sum()
    # and a list comprehension to try all permutations.
    
    # We use a list of all pairs of indices (i, j) where 1 <= i < j <= N
    indices_pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    
    # For a given permutation P (which is a tuple), the cost is:
    # sum(cost_map[tuple(sorted((P[i-1], P[j-1])))] 
    #     for (i, j) in indices_pairs 
    #     if ((i, j) in G_edges) != (tuple(sorted((P[i-1], P[j-1]))) in H_edges))
    
    # We wrap this in a min() over all permutations.
    # Note: G_edges and H_edges already contain sorted tuples.
    
    ans = min(
        sum(
            cost_map[tuple(sorted((p[i-1], p[j-1])))]
            for (i, j) in indices_pairs
            if ((i, j) in G_edges) != (tuple(sorted((p[i-1], p[j-1]))) in H_edges)
        )
        for p in permutations(range(1, N + 1))
    )
    
    print(ans)

if __name__ == "__main__":
    solve()