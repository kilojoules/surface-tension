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
    
    # Helper to get edge presence as a boolean matrix
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
    
    # Read cost matrix A
    # A is given as a flattened upper triangle
    costs = [int(next(it)) for _ in range(N * (N - 1) // 2)]
    
    # Map (i, j) where i < j to the index in the costs list
    # Index = (sum of (N-1) + (N-2) ... + (N-i)) + (j-1)
    # Or more simply, we can pre-calculate the cost matrix
    cost_matrix = [[0] * N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            cost_matrix[i][j] = cost_matrix[j][i] = costs[idx]
            idx += 1

    # The goal is to find a permutation P of {0, ..., N-1} that minimizes:
    # Sum_{i < j} cost_matrix[P[i]][P[j]] if (adj_G[i][j] != adj_H[P[i]][P[j]])
    
    # Generate all permutations of (0, ..., N-1)
    # Note: N <= 8, so N! is 40,320, which is feasible.
    all_perms = permutations(range(N))
    
    # We want to calculate the cost for each permutation and find the minimum.
    # To avoid loops, we use map and a helper function.
    
    def calculate_cost(p):
        # p is the permutation where vertex i in G maps to vertex p[i] in H
        # Total cost = sum_{0 <= i < j < N} cost(p[i], p[j]) if G(i,j) != H(p[i], p[j])
        return sum(
            cost_matrix[p[i]][p[j]] 
            for i in range(N) 
            for j in range(i + 1, N) 
            if adj_G[i][j] != adj_H[p[i]][p[j]]
        )

    # Use min() with a generator to find the minimum cost
    # This avoids explicit for/while loops.
    result = min(map(calculate_cost, all_perms))
    print(result)

if __name__ == "__main__":
    solve()