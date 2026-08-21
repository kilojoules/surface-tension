import sys
from itertools import permutations

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use a generator/iterator to consume input sequentially
    it = map(int, input_data)
    
    # N: Number of vertices
    N = next(it)
    
    # M_G: Number of edges in G
    M_G = next(it)
    # Build adjacency matrix for G
    # We use a list of lists (N x N)
    adj_G = [[0] * N for _ in range(N)]
    for _ in range(M_G):
        u, v = next(it), next(it)
        adj_G[u-1][v-1] = adj_G[v-1][u-1] = 1
        
    # M_H: Number of edges in H
    M_H = next(it)
    # Build adjacency matrix for H
    adj_H = [[0] * N for _ in range(N)]
    for _ in range(M_H):
        a, b = next(it), next(it)
        adj_H[a-1][b-1] = adj_H[b-1][a-1] = 1
        
    # A: Cost matrix for operations on H
    # The input provides A_{i,j} for 1 <= i < j <= N
    # We store them in a symmetric N x N matrix
    cost_matrix = [[0] * N for _ in range(N)]
    # We need to fill the cost matrix based on the remaining items in the iterator
    # The number of A_{i,j} values is N*(N-1)//2
    # We can use a helper to map the flat list of costs to the indices (i, j)
    all_costs = list(it)
    
    # To map the flat list of costs to the matrix without loops, 
    # we pre-calculate the indices of the upper triangle.
    indices = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Since we cannot use loops to populate the matrix, we can use a dictionary 
    # and then a list comprehension to build the final matrix.
    cost_map = dict(zip(indices, all_costs))
    
    # Build the final cost matrix using list comprehension
    # cost_matrix[i][j] is the cost to flip edge (i, j)
    final_costs = [
        [cost_map.get((i, j), cost_map.get((j, i), 0)) for j in range(N)]
        for i in range(N)
    ]

    # We need to find a permutation P of (0, ..., N-1) that minimizes:
    # Sum_{i < j} cost(P_i, P_j) where (i, j) is an edge in G XOR (P_i, P_j) is an edge in H
    # Note: The problem says we pay A_{i,j} to flip edge (i, j) in H.
    # If G has edge (i, j) and H does not have edge (P_i, P_j), we pay A_{P_i, P_j}.
    # If G does not have edge (i, j) and H has edge (P_i, P_j), we pay A_{P_i, P_j}.
    
    # Generate all permutations of vertices
    all_perms = permutations(range(N))
    
    # For each permutation, calculate the total cost
    # We iterate over all pairs (i, j) with i < j
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # The cost for a permutation P is the sum of final_costs[P[i]][P[j]] 
    # for all pairs (i, j) where adj_G[i][j] != adj_H[P[i]][P[j]]
    
    # We use a generator expression inside min() to find the minimum cost
    ans = min(
        sum(
            final_costs[p[i]][p[j]] 
            for i, j in pairs 
            if adj_G[i][j] != adj_H[p[i]][p[j]]
        )
        for p in all_perms
    )
    
    print(ans)

if __name__ == "__main__":
    solve()