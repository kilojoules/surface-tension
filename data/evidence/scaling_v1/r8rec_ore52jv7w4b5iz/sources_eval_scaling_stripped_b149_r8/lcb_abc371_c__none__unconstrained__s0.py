import itertools
import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    ptr = 0
    N = int(input_data[ptr])
    ptr += 1
    
    M_G = int(input_data[ptr])
    ptr += 1
    
    # Build adjacency matrix for G
    # Using a list of lists for G since it's static
    adj_G = [[0] * N for _ in range(N)]
    for _ in range(M_G):
        u = int(input_data[ptr]) - 1
        v = int(input_data[ptr+1]) - 1
        adj_G[u][v] = adj_G[v][u] = 1
        ptr += 2
        
    M_H = int(input_data[ptr])
    ptr += 1
    
    # Build adjacency matrix for H
    adj_H = [[0] * N for _ in range(N)]
    for _ in range(M_H):
        u = int(input_data[ptr]) - 1
        v = int(input_data[ptr+1]) - 1
        adj_H[u][v] = adj_H[v][u] = 1
        ptr += 2
        
    # Read costs A_{i,j} into a 2D array
    # The input provides A_{i,j} for i < j
    # We'll store them in a symmetric matrix for easy access
    costs_flat = list(map(int, input_data[ptr:]))
    
    # To map the flat list of costs to A[i][j], we need to know the indices.
    # There are N*(N-1)//2 costs.
    # We can reconstruct the cost matrix by iterating through i < j.
    cost_matrix = [[0] * N for _ in range(N)]
    
    # Use a generator to consume the flat list
    cost_gen = iter(costs_flat)
    for i in range(N):
        for j in range(i + 1, N):
            c = next(cost_gen)
            cost_matrix[i][j] = cost_matrix[j][i] = c

    # We need to find a permutation P of (0, ..., N-1) that minimizes:
    # Sum_{i < j} cost(P_i, P_j) if (edge(i, j) in G != edge(P_i, P_j) in H)
    
    # Pre-calculate all pairs (i, j) where i < j
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Generate all permutations of 0...N-1
    perms = itertools.permutations(range(N))
    
    # For each permutation P, calculate the total cost.
    # The cost is incurred if the edge status between i and j in G 
    # differs from the edge status between P[i] and P[j] in H.
    # Note: The operation is performed on H to make it isomorphic to G.
    # The cost A_{i,j} is associated with vertices i and j of graph H.
    # So if G has an edge (i, j) and H doesn't have an edge (P_i, P_j),
    # we pay A_{P_i, P_j} to add it.
    
    def calculate_cost(p):
        # p is the permutation such that vertex i in G maps to vertex p[i] in H
        return sum(
            cost_matrix[p[i]][p[j]]
            for i, j in pairs
            if adj_G[i][j] != adj_H[p[i]][p[j]]
        )

    # Find the minimum cost across all permutations
    # Using map and min to avoid explicit for loops
    ans = min(map(calculate_cost, perms))
    print(ans)

if __name__ == "__main__":
    solve()