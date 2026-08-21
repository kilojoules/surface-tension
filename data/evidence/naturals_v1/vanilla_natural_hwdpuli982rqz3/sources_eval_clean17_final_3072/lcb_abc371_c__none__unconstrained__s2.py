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
    
    # Helper to get edge presence as a bitmask or tuple
    # Since N is small (up to 8), we can represent the graph as an adjacency matrix
    # or a set of edges.
    
    # Read G edges
    MG = int(next(it))
    G_edges = [tuple(map(int, (next(it), next(it)))) for _ in range(MG)]
    
    # Read H edges
    MH = int(next(it))
    H_edges = [tuple(map(int, (next(it), next(it)))) for _ in range(MH)]
    
    # Read A costs into a 2D structure
    # A[i][j] where 1 <= i < j <= N
    # We'll store it in a dictionary or a nested list
    costs = {}
    for i in range(1, N):
        row_costs = [int(next(it)) for _ in range(N - i)]
        for j_idx, cost in enumerate(row_costs):
            costs[(i, i + 1 + j_idx)] = cost

    # Precompute adjacency matrices for G and H
    # Using 0-indexed internally
    adj_G = [[0] * N for _ in range(N)]
    for u, v in G_edges:
        adj_G[u-1][v-1] = adj_G[v-1][u-1] = 1
        
    adj_H = [[0] * N for _ in range(N)]
    for u, v in H_edges:
        adj_H[u-1][v-1] = adj_H[v-1][u-1] = 1

    # The cost to make H isomorphic to G under permutation P is:
    # Sum_{1 <= i < j <= N} cost(P_i, P_j) if (edge(i, j) in G != edge(P_i, P_j) in H)
    # Where P is a permutation of {0, ..., N-1}
    
    # To minimize memory/loops, we pre-calculate the cost matrix for H
    # cost_matrix[i][j] = A_{i+1, j+1}
    cost_mat = [[0] * N for _ in range(N)]
    for (i, j), c in costs.items():
        cost_mat[i-1][j-1] = cost_mat[j-1][i-1] = c

    # We iterate through all permutations P of (0, ..., N-1)
    # P[i] is the vertex in H that corresponds to vertex i in G.
    # Edge (i, j) in G exists <=> Edge (P[i], P[j]) in H exists.
    
    def calculate_cost(p):
        # p is the permutation tuple
        # We need to sum cost_mat[p[i]][p[j]] for all i < j where adj_G[i][j] != adj_H[p[i]][p[j]]
        return sum(
            cost_mat[p[i]][p[j]] 
            for i in range(N) 
            for j in range(i + 1, N) 
            if adj_G[i][j] != adj_H[p[i]][p[j]]
        )

    # Use map to apply the calculation to all permutations and find the minimum
    # permutations(range(N)) is a generator, map processes it, min finds the minimum
    ans = min(map(calculate_cost, permutations(range(N))))
    print(ans)

if __name__ == "__main__":
    solve()