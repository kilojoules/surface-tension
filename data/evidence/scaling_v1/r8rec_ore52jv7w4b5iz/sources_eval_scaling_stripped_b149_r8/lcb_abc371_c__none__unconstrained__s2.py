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
    G_edges = []
    for _ in range(MG):
        u = int(input_data[ptr])
        v = int(input_data[ptr+1])
        G_edges.append(tuple(sorted((u, v))))
        ptr += 2
        
    # Graph H edges
    MH = int(input_data[ptr])
    ptr += 1
    H_edges = []
    for _ in range(MH):
        u = int(input_data[ptr])
        v = int(input_data[ptr+1])
        H_edges.append(tuple(sorted((u, v))))
        ptr += 2
        
    # Cost matrix A
    # A[i][j] will store cost for pair (i+1, j+1)
    # The input provides A_{1,2}, A_{1,3}... A_{1,N}, then A_{2,3}...
    # We flatten the remaining input into a list
    flat_A = list(map(int, input_data[ptr:]))
    
    # To map (i, j) where 1 <= i < j <= N to the flat_A index:
    # The number of pairs (1, k) is N-1
    # The number of pairs (2, k) is N-2
    # Index for (i, j) is sum_{k=1}^{i-1} (N-k) + (j-i-1)
    # But it's easier to just build a dictionary or a 2D array.
    
    # Reconstructing A into a 2D array for O(1) access
    # A_matrix[i][j] is cost for edge between vertex i+1 and j+1
    A_matrix = [[0] * N for _ in range(N)]
    curr = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            A_matrix[i][j] = A_matrix[j][i] = flat_A[curr]
            curr += 1

    # Adjacency matrices for G and H
    adj_G = [[False] * N for _ in range(N)]
    for u, v in G_edges:
        adj_G[u-1][v-1] = adj_G[v-1][u-1] = True
        
    adj_H = [[False] * N for _ in range(N)]
    for u, v in H_edges:
        adj_H[u-1][v-1] = adj_H[v-1][u-1] = True

    # We need to find a permutation P of {0, ..., N-1} that minimizes:
    # Sum_{i < j} A[P[i]][P[j]] if (adj_G[i][j] != adj_H[P[i]][P[j]])
    
    # Pre-calculate all possible pairs (i, j) with i < j
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Use a generator expression inside min() to find the minimum cost across all permutations
    # P is the mapping from G-vertices to H-vertices
    ans = min(
        sum(
            A_matrix[p[i]][p[j]] 
            for i, j in pairs 
            if adj_G[i][j] != adj_H[p[i]][p[j]]
        )
        for p in permutations(range(N))
    )
    
    print(ans)

if __name__ == "__main__":
    solve()