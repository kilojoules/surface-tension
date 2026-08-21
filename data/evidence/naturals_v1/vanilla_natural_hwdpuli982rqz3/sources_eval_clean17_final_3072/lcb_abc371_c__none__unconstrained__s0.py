import sys
from itertools import permutations
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use an iterator to handle the input stream
    it = iter(input_data)
    
    N = int(next(it))
    
    # Helper to get edge existence as a boolean matrix
    def get_adj_matrix(m_count, iterator):
        adj = [[False] * N for _ in range(N)]
        for _ in range(m_count):
            u = int(next(iterator)) - 1
            v = int(next(iterator)) - 1
            adj[u][v] = adj[v][u] = True
        return adj

    M_G = int(next(it))
    adj_G = get_adj_matrix(M_G, it)
    
    M_H = int(next(it))
    adj_H = get_adj_matrix(M_H, it)
    
    # Read costs A_{i,j} into a matrix
    # The input format for A is a flattened upper triangle
    costs = [[0] * N for _ in range(N)]
    for i in range(N - 1):
        for j in range(i + 1, N):
            costs[i][j] = costs[j][i] = int(next(it))

    # Precompute all pairs (i, j) where i < j
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    # The goal is to find a permutation P of {0, ..., N-1} that minimizes:
    # Sum_{i < j} cost(i, j) if (edge (i, j) in G != edge (P_i, P_j) in H)
    
    # We use map() and min() to avoid explicit for/while loops
    # P is the permutation. P[i] is the vertex in H corresponding to vertex i in G.
    
    def calculate_cost(P):
        # For each pair of vertices (i, j) in G, check if the edge status 
        # matches the pair (P[i], P[j]) in H.
        # If they differ, we must perform an operation on H at cost A_{P[i], P[j]}.
        return sum(
            costs[P[i]][P[j]] 
            for i, j in pairs 
            if adj_G[i][j] != adj_H[P[i]][P[j]]
        )

    # Generate all permutations of range(N) and find the minimum cost
    # We use a generator expression inside min() to satisfy constraints
    ans = min(map(calculate_cost, permutations(range(N))))
    
    print(ans)

if __name__ == "__main__":
    solve()