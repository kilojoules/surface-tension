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
    g_edges = []
    for _ in range(MG):
        g_edges.append((int(input_data[ptr]), int(input_data[ptr+1])))
        ptr += 2
        
    # Graph H edges
    MH = int(input_data[ptr])
    ptr += 1
    h_edges = []
    for _ in range(MH):
        h_edges.append((int(input_data[ptr]), int(input_data[ptr+1])))
        ptr += 2
        
    # Cost matrix A
    # A is given as a flattened upper triangle
    a_flat = [int(x) for x in input_data[ptr:]]
    
    # Pre-calculate adjacency matrices for G and H
    # Using 0-indexing internally
    adj_g = [[0] * N for _ in range(N)]
    [adj_g[u-1].__setitem__(v-1, 1) or adj_g[v-1].__setitem__(u-1, 1) for u, v in g_edges]
    
    adj_h = [[0] * N for _ in range(N)]
    [adj_h[u-1].__setitem__(v-1, 1) or adj_h[v-1].__setitem__(u-1, 1) for u, v in h_edges]
    
    # Map the flattened A matrix to a 2D lookup
    # A_{i,j} is provided for 1 <= i < j <= N
    # We create a symmetric matrix for easy access
    cost_matrix = [[0] * N for _ in range(N)]
    
    # To populate cost_matrix without loops, we use a helper to map indices
    # The number of elements in A is N*(N-1)//2
    # We can use a list comprehension to trigger the sets
    def get_cost_coords():
        # Generates (i, j) for the upper triangle
        return [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    coords = get_cost_coords()
    # Use a list comprehension to fill the cost matrix
    [cost_matrix[coords[k][0]].__setitem__(coords[k][1], a_flat[k]) or 
     cost_matrix[coords[k][1]].__setitem__(coords[k][0], a_flat[k]) 
     for k in range(len(a_flat))]

    # We need to find a permutation P of (0, ..., N-1) that minimizes:
    # Sum_{i < j} cost(i, j) if (edge(i, j) in G != edge(P_i, P_j) in H)
    
    # Generate all permutations of vertices
    all_perms = permutations(range(N))
    
    # For a fixed permutation P, the cost is:
    # Sum_{i < j} cost_matrix[P_i][P_j] where adj_g[i][j] != adj_h[P_i][P_j]
    
    # To avoid loops, we use a nested comprehension:
    # Outer: permutations
    # Inner: sum of costs for all pairs (i, j)
    
    # Pre-calculate the pairs (i, j) to avoid recreating them
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # The final answer is the minimum cost over all permutations
    ans = min([
        sum([
            cost_matrix[p[i]][p[j]] 
            for i, j in pairs 
            if adj_g[i][j] != adj_h[p[i]][p[j]]
        ])
        for p in all_perms
    ])
    
    print(ans)

if __name__ == "__main__":
    solve()