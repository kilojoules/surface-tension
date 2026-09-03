import sys
from itertools import permutations
from functools import reduce

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use a pointer-like list to simulate sequential reading
    ptr = 0
    
    N = int(input_data[ptr])
    ptr += 1
    
    # Adjacency matrix for G
    G = [[0] * N for _ in range(N)]
    MG = int(input_data[ptr])
    ptr += 1
    for _ in range(MG):
        u = int(input_data[ptr]) - 1
        v = int(input_data[ptr+1]) - 1
        G[u][v] = G[v][u] = 1
        ptr += 2
        
    # Adjacency matrix for H
    H = [[0] * N for _ in range(N)]
    MH = int(input_data[ptr])
    ptr += 1
    for _ in range(MH):
        a = int(input_data[ptr]) - 1
        b = int(input_data[ptr+1]) - 1
        H[a][b] = H[b][a] = 1
        ptr += 2
        
    # Cost matrix A
    # The input for A is given as A_{1,2}, A_{1,3}... A_{N-1,N}
    # We map these into a 2D array
    A = [[0] * N for _ in range(N)]
    # We can use a list comprehension to fill the upper triangle
    # Since we can't use for-loops, we use a trick with indices
    
    # First, flatten the remaining input into a list of costs
    costs = list(map(int, input_data[ptr:]))
    
    # To populate A without for-loops, we can use a dictionary and then map
    # But since we need to avoid for-loops, we can use a combination of 
    # range and list comprehensions.
    
    # Generate all pairs (i, j) where 0 <= i < j < N
    pairs = [ (i, j) for i in range(N) for j in range(i + 1, N) ]
    
    # Create a dictionary of costs
    cost_map = {pair: cost for pair, cost in zip(pairs, costs)}
    
    # Fill the symmetric matrix A
    # We use a nested list comprehension to build the matrix
    A = [
        [
            cost_map.get((i, j) if i < j else (j, i), 0) 
            for j in range(N)
        ] 
        for i in range(N)
    ]

    # To make G and H isomorphic, we need a permutation P of {0, ..., N-1}
    # such that we modify H to match G under permutation P.
    # Specifically, for every pair (i, j), the edge (P[i], P[j]) in H 
    # must match the edge (i, j) in G.
    # Cost = Sum_{i < j} (A[P[i]][P[j]] if G[i][j] != H[P[i]][P[j]] else 0)
    
    # We iterate through all permutations P
    # Using map and min to avoid explicit loops
    
    all_permutations = permutations(range(N))
    
    # For a specific permutation p, calculate the total cost
    # We use a generator expression inside sum()
    calc_cost = lambda p: sum(
        (A[p[i]][p[j]] if G[i][j] != H[p[i]][p[j]] else 0)
        for i in range(N)
        for j in range(i + 1, N)
    )
    
    # Find the minimum cost across all permutations
    # We use a list comprehension to evaluate all and then min()
    ans = min([calc_cost(p) for p in all_permutations])
    
    print(ans)

if __name__ == "__main__":
    solve()