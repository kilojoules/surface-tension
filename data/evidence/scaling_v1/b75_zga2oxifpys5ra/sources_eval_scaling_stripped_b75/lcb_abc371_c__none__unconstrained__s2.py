import itertools
import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use a generator/iterator to consume input
    it = iter(input_data)
    
    # N: Number of vertices
    N = int(next(it))
    
    # M_G: Number of edges in G
    M_G = int(next(it))
    # G edges: use a set of frozen sets for undirected edges
    G_edges = {frozenset([int(next(it)), int(next(it))]) for _ in range(M_G)}
    
    # M_H: Number of edges in H
    M_H = int(next(it))
    # H edges: use a set of frozen sets
    H_edges = {frozenset([int(next(it)), int(next(it))]) for _ in range(M_H)}
    
    # A_{i,j}: Costs of toggling edges
    # The input provides A_{i,j} for 1 <= i < j <= N
    # We store them in a dictionary mapping frozenset({i, j}) to cost
    # There are N*(N-1)//2 such values
    costs_list = [int(next(it)) for _ in range(N * (N - 1) // 2)]
    
    # Generate all pairs (i, j) with 1 <= i < j <= N in the order they appear in input
    all_pairs = [frozenset([i, j]) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    A = dict(zip(all_pairs, costs_list))
    
    # We need to find a permutation P of (1, ..., N) that minimizes the cost
    # Cost for a permutation P:
    # Sum over all pairs {i, j} of A_{P_i, P_j} if (edge {i, j} in G) != (edge {P_i, P_j} in H)
    
    # Pre-calculate which pairs in G have edges
    # We use a list of all possible pairs to avoid loops in the comprehension
    g_edge_status = {pair: (pair in G_edges) for pair in all_pairs}
    
    # Try all permutations of (1, ..., N)
    # For each permutation P, calculate the total cost
    # P maps vertex i in G to vertex P[i-1] in H
    
    # To optimize, we can pre-calculate the cost for every pair of vertices in H
    # given whether they should have an edge or not.
    # But with N=8, N! is 40,320, which is small enough for a comprehension.
    
    # We use a helper to get the cost of a pair under permutation P
    # Pair {i, j} in G corresponds to pair {P[i-1], P[j-1]} in H
    
    # To make it fast, we pre-calculate the adjacency matrix of H
    # H_adj[i][j] is True if edge {i, j} exists in H
    H_adj = [[False] * (N + 1) for _ in range(N + 1)]
    for edge in H_edges:
        u, v = list(edge)
        H_adj[u][v] = H_adj[v][u] = True

    # We pre-calculate the cost of toggling edge {u, v} in H
    # If G says there should be an edge and H doesn't, or vice versa, pay A_{u,v}
    
    # Let's define a function that calculates cost for a permutation P
    # P is a tuple (P1, P2, ..., PN)
    # For all 1 <= i < j <= N:
    # if (edge {i, j} in G) != (edge {P_i, P_j} in H), add A_{P_i, P_j}
    
    # To avoid loops, we use a generator expression inside sum()
    # We iterate over all pairs (i, j) with 1 <= i < j <= N
    
    # Pre-calculating G's edge existence for all pairs
    # pairs_g is a list of (i, j, exists_in_g)
    pairs_g = [(i, j, frozenset([i, j]) in G_edges) 
                for i in range(1, N + 1) 
                for j in range(i + 1, N + 1)]

    # The core logic:
    # For a permutation P, the cost is:
    # sum(A[frozenset([P[i-1], P[j-1]])] 
    #     for (i, j, in_g) in pairs_g 
    #     if in_g != H_adj[P[i-1]][P[j-1]])
    
    # We use map and lambda to avoid 'for' loops and maintain "functional" style
    # though a comprehension is essentially the same.
    
    ans = min(
        sum(A[frozenset([p[i-1], p[j-1]])] 
            for (i, j, in_g) in pairs_g 
            if in_g != H_adj[p[i-1]][p[j-1]])
        for p in itertools.permutations(range(1, N + 1))
    )
    
    print(ans)

if __name__ == "__main__":
    solve()