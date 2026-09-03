import sys
from itertools import permutations
from functools import reduce

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use an iterator to handle the input stream
    it = iter(input_data)
    
    N = int(next(it))
    
    # Helper to get edge presence as a boolean matrix
    def get_adj_matrix(num_edges, iterator):
        adj = [[False] * N for _ in range(N)]
        for _ in range(num_edges):
            u = int(next(iterator)) - 1
            v = int(next(iterator)) - 1
            adj[u][v] = adj[v][u] = True
        return adj

    # Read Graph G
    MG = int(next(it))
    adj_G = get_adj_matrix(MG, it)
    
    # Read Graph H
    MH = int(next(it))
    adj_H = get_adj_matrix(MH, it)
    
    # Read Costs A into a 2D list
    # The input format for A is a flattened triangle
    costs_flat = [int(next(it)) for _ in range(N * (N - 1) // 2)]
    
    # Map (i, j) where i < j to the cost in costs_flat
    # Index = (i * (2*N - 1 - i) // 2) + (j - i - 1)
    # However, since N is small (8), we can just build the matrix
    adj_A = [[0] * N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            adj_A[i][j] = adj_A[j][i] = costs_flat[idx]
            idx += 1

    # Precompute all pairs (i, j) with i < j to avoid nested loops in lambda
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    # For a given permutation P, the cost to make H isomorphic to G is:
    # Sum_{i < j} cost(P_i, P_j) if (edge(i, j) in G != edge(P_i, P_j) in H)
    # Note: The problem defines P such that edge(i, j) in G <=> edge(P_i, P_j) in H.
    # Let's use a permutation P where P[i] is the vertex in H corresponding to vertex i in G.
    
    calc_cost = lambda p: sum(
        adj_A[p[i]][p[j]] for (i, j) in pairs 
        if adj_G[i][j] != adj_H[p[i]][p[j]]
    )

    # Iterate through all permutations of (0, ..., N-1) and find the minimum cost
    # Using map and min to avoid explicit for/while loops
    ans = min(map(calc_cost, permutations(range(N))))
    
    print(ans)

if __name__ == "__main__":
    solve()