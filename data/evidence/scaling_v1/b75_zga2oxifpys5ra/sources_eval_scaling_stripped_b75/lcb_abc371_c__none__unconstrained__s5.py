import sys
from itertools import permutations

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use a generator/iterator to consume input
    it = iter(input_data)
    
    # N is the number of vertices
    N = int(next(it))
    
    # M_G is the number of edges in G
    M_G = int(next(it))
    # G edges: use a set of frozensets for fast lookup
    # We use 0-indexing internally, so subtract 1 from vertex indices
    G_edges = {frozenset([int(next(it)) - 1, int(next(it)) - 1]) for _ in range(M_G)}
    
    # M_H is the number of edges in H
    M_H = int(next(it))
    # H edges: use a set of frozensets
    H_edges = {frozenset([int(next(it)) - 1, int(next(it)) - 1]) for _ in range(M_H)}
    
    # A_{i,j} costs
    # The input provides A_{i,j} for 1 <= i < j <= N
    # We store them in a 2D list where cost[i][j] is the cost to flip edge (i, j)
    # Since N is small (<= 8), we can use a nested list
    costs_flat = [int(next(it)) for _ in range(N * (N - 1) // 2)]
    
    # Map the flat cost list to a 2D array for O(1) access
    # cost_matrix[i][j] will store A_{i+1, j+1}
    cost_matrix = [[0] * N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            cost_matrix[i][j] = cost_matrix[j][i] = costs_flat[idx]
            idx += 1

    # Pre-calculate all possible edges in a graph of size N
    all_edges = [frozenset([i, j]) for i in range(N) for j in range(i + 1, N)]

    # We need to find a permutation P of {0, ..., N-1} that minimizes the cost
    # The cost for a permutation P is the sum of cost_matrix[P_i][P_j] 
    # for all pairs (i, j) where the edge status in G(i, j) differs from H(P_i, P_j)
    
    # To optimize, we can pre-determine for every pair (i, j) if G has an edge
    g_edge_exists = {edge: (edge in G_edges) for edge in all_edges}
    
    # We iterate through all N! permutations
    # For each permutation P, we calculate the total cost
    # Cost = sum(cost_matrix[P[i]][P[j]] if G_edge(i,j) != H_edge(P[i], P[j]))
    
    # Using a generator expression inside min() to avoid explicit loops
    # We use a helper logic: (edge_in_G != edge_in_H) is 1 if they differ, 0 otherwise
    
    # To make it faster, we pre-calculate the adjacency of H
    h_adj = [[(frozenset([i, j]) in H_edges) for j in range(N)] for i in range(N)]
    g_adj = [[(frozenset([i, j]) in G_edges) for j in range(N)] for i in range(N)]

    # The cost for a permutation P is:
    # sum_{i<j} cost_matrix[P[i]][P[j]] * (g_adj[i][j] ^ h_adj[P[i]][P[j]])
    
    # We use a list comprehension to generate costs for all permutations
    # and then find the minimum.
    ans = min(
        sum(
            cost_matrix[p[i]][p[j]] 
            for i in range(N) 
            for j in range(i + 1, N) 
            if g_adj[i][j] != h_adj[p[i]][p[j]]
        )
        for p in permutations(range(N))
    )
    
    print(ans)

if __name__ == "__main__":
    solve()