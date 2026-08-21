import sys
from itertools import permutations

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    ptr = 0
    N = int(input_data[ptr])
    ptr += 1
    
    # Graph G edges
    MG = int(input_data[ptr])
    ptr += 1
    g_edges_list = []
    for _ in range(MG):
        g_edges_list.append((int(input_data[ptr]), int(input_data[ptr+1])))
        ptr += 2
        
    # Graph H edges
    MH = int(input_data[ptr])
    ptr += 1
    h_edges_list = []
    for _ in range(MH):
        h_edges_list.append((int(input_data[ptr]), int(input_data[ptr+1])))
        ptr += 2
        
    # Cost matrix A
    # A[i][j] will store cost for pair (i+1, j+1)
    # The input provides A_{1,2}, A_{1,3}... A_{N-1,N}
    # We flatten the remaining input and map it to a dictionary for easy access
    costs_flat = list(map(int, input_data[ptr:]))
    
    # Pre-calculate adjacency matrices for G and H
    # Using 0-indexing internally
    adj_g = [[0] * N for _ in range(N)]
    [adj_g[u-1].__setitem__(v-1, 1) or adj_g[v-1].__setitem__(u-1, 1) for u, v in g_edges_list]
    
    adj_h = [[0] * N for _ in range(N)]
    [adj_h[u-1].__setitem__(v-1, 1) or adj_h[v-1].__setitem__(u-1, 1) for u, v in h_edges_list]
    
    # Map the flat cost list to a 2D array A[i][j] where 0 <= i < j < N
    # The input order is A_{1,2}, A_{1,3}... A_{1,N}, A_{2,3}...
    # We can reconstruct this by iterating through the indices
    cost_matrix = [[0] * N for _ in range(N)]
    
    # To avoid loops, we use a list comprehension to populate the cost matrix
    # We create a list of all pairs (i, j) in the order they appear in the input
    pairs = [(i, j) for i in range(N-1) for j in range(i+1, N)]
    # Use a side-effect in a list comprehension to fill the matrix
    [cost_matrix[p[0]].__setitem__(p[1], costs_flat[idx]) or cost_matrix[p[1]].__setitem__(p[0], costs_flat[idx]) 
     for idx, p in enumerate(pairs)]

    # We want to find a permutation P of {0, ..., N-1} such that 
    # the cost to make H match G under mapping P is minimized.
    # Cost for pair (i, j) is A_{P_i, P_j} if (i, j) is an edge in G 
    # but not in H, or vice versa.
    
    # Pre-calculate all pairs (i, j) with i < j
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # The cost for a specific permutation P is:
    # Sum_{i < j} (Cost(P_i, P_j) if (AdjG[i][j] != AdjH[P_i][P_j]) else 0)
    
    # We use min() over all permutations
    ans = min(
        sum(
            cost_matrix[p[i]][p[j]] 
            for i, j in all_pairs 
            if adj_g[i][j] != adj_h[p[i]][p[j]]
        )
        for p in permutations(range(N))
    )
    
    print(ans)

if __name__ == "__main__":
    solve()