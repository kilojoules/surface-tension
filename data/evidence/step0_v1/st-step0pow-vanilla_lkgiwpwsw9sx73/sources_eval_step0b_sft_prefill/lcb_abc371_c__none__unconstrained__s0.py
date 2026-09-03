import sys
from itertools import permutations

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
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
    # The input gives A_{i,j} for 1 <= i < j <= N
    A = [[0] * N for _ in range(N)]
    for i in range(N - 1):
        for j in range(i + 1, N):
            A[i][j] = A[j][i] = int(input_data[ptr])
            ptr += 1

    # To make G and H isomorphic, we need a permutation P of {0, ..., N-1}
    # such that we modify H to match G under mapping P.
    # Specifically, for every pair (i, j), the edge (P[i], P[j]) in H 
    # must be the same as the edge (i, j) in G.
    # Cost = sum_{i < j} (A[P[i]][P[j]] if G[i][j] != H[P[i]][P[j]] else 0)
    
    # We iterate through all permutations P of range(N)
    # Using a generator expression inside min() to avoid explicit for-loops
    # and map() to handle the cost calculation for each permutation.
    
    def calculate_cost(p):
        # p is the permutation tuple
        # We sum costs for all pairs (i, j) where 0 <= i < j < N
        return sum(
            A[p[i]][p[j]] 
            for i in range(N) 
            for j in range(i + 1, N) 
            if G[i][j] != H[p[i]][p[j]]
        )

    # Find the minimum cost across all permutations
    # permutations(range(N)) is a generator, map applies the cost function
    ans = min(map(calculate_cost, permutations(range(N))))
    print(ans)

if __name__ == "__main__":
    solve()