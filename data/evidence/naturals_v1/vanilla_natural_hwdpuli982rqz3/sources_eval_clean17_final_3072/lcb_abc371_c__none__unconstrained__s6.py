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
    
    # N is the number of vertices
    N = int(next(it))
    
    # Helper to get edge status as a boolean matrix
    def get_adj_matrix(m_count):
        adj = [[False] * N for _ in range(N)]
        for _ in range(m_count):
            u = int(next(it)) - 1
            v = int(next(it)) - 1
            adj[u][v] = adj[v][u] = True
        return adj

    # Read G and H adjacency matrices
    MG = int(next(it))
    adj_G = get_adj_matrix(MG)
    MH = int(next(it))
    adj_H = get_adj_matrix(MH)
    
    # Read cost matrix A
    # The input for A is provided as a flattened triangle
    costs = [int(x) for x in it]
    
    # Map the flattened costs list back to a 2D matrix A[i][j]
    # A_flat is A[0,1], A[0,2]...A[0,N-1], A[1,2]...
    A = [[0] * N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            A[i][j] = A[j][i] = costs[idx]
            idx += 1

    # Precompute all pairs (i, j) where i < j
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    # The goal is to find a permutation P of (0, ..., N-1) that minimizes:
    # Sum_{i < j} cost_to_match(G_edge(i, j), H_edge(P_i, P_j))
    # where cost_to_match is A[P_i][P_j] if the edge statuses differ, else 0.
    
    # To avoid loops and recursion, we use a generator expression inside min()
    # We iterate through all permutations of range(N)
    
    ans = min(
        sum(
            A[p[i]][p[j]] if adj_G[i][j] != adj_H[p[i]][p[j]] else 0
            for i, j in pairs
        )
        for p in permutations(range(N))
    )
    
    print(ans)

if __name__ == "__main__":
    solve()