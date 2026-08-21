import sys
from itertools import permutations

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to consume the input stream
    it = iter(input_data)
    
    N = next(it)
    
    # Graph G edges
    MG = next(it)
    G_edges = [tuple(sorted((next(it), next(it)))) for _ in range(MG)]
    
    # Graph H edges
    MH = next(it)
    H_edges = [tuple(sorted((next(it), next(it)))) for _ in range(MH)]
    
    # Cost matrix A
    # A[i][j] will store the cost to flip edge (i+1, j+1)
    # The input provides A_{1,2}, A_{1,3}... A_{N-1,N}
    # We build a 2D list where A[i][j] is the cost for vertices i+1 and j+1
    # Since N is small (<= 8), we can afford a full matrix
    costs_flat = [next(it) for _ in range(N * (N - 1) // 2)]
    
    # To map the flat cost list to a 2D matrix A[i][j]
    # We use a dictionary or a matrix. Let's use a matrix for O(1) access.
    # We need to fill the matrix based on the input order.
    # The input order is (1,2), (1,3)...(1,N), (2,3)...(2,N), etc.
    
    # Create a mapping of (i, j) pairs to their index in the costs_flat list
    pairs = [ (i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1) ]
    cost_map = {pair: cost for pair, cost in zip(pairs, costs_flat)}

    # Adjacency matrices for G and H for O(1) edge lookup
    adj_G = [[False] * (N + 1) for _ in range(N + 1)]
    for u, v in G_edges:
        adj_G[u][v] = adj_G[v][u] = True
        
    adj_H = [[False] * (N + 1) for _ in range(N + 1)]
    for u, v in H_edges:
        adj_H[u][v] = adj_H[v][u] = True

    # We want to find a permutation P of (1...N) such that 
    # the cost to make H isomorphic to G is minimized.
    # The cost for a permutation P is the sum of A_{P_i, P_j} 
    # for all pairs (i, j) where (i, j) is an edge in G 
    # but (P_i, P_j) is NOT an edge in H, OR vice versa.
    
    # Generate all permutations of (1...N)
    all_perms = permutations(range(1, N + 1))
    
    # For each permutation, calculate the total cost
    # We iterate over all pairs 1 <= i < j <= N
    # If adj_G[i][j] != adj_H[P[i-1]][P[j-1]], we add the cost A_{P[i-1], P[j-1]}
    
    # To avoid loops, we use a generator expression inside min()
    # We use a helper to get the cost from the cost_map
    
    ans = min(
        sum(
            cost_map[tuple(sorted((p[i-1], p[j-1]))))]
            for i in range(1, N + 1)
            for j in range(i + 1, N + 1)
            if adj_G[i][j] != adj_H[p[i-1]][p[j-1]]
        )
        for p in all_perms
    )
    
    print(ans)

if __name__ == "__main__":
    solve()