import sys
from itertools import permutations

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use a pointer-like list to simulate sequential reading
    ptr = 0
    
    N = int(input_data[ptr])
    ptr += 1
    
    # Adjacency matrices for G and H
    # Using 0-indexed internally
    adj_G = [[0] * N for _ in range(N)]
    
    M_G = int(input_data[ptr])
    ptr += 1
    for _ in range(M_G):
        u = int(input_data[ptr]) - 1
        v = int(input_data[ptr+1]) - 1
        adj_G[u][v] = adj_G[v][u] = 1
        ptr += 2
        
    M_H = int(input_data[ptr])
    ptr += 1
    adj_H = [[0] * N for _ in range(N)]
    for _ in range(M_H):
        a = int(input_data[ptr]) - 1
        b = int(input_data[ptr+1]) - 1
        adj_H[a][b] = adj_H[b][a] = 1
        ptr += 2
        
    # Cost matrix A
    # A[i][j] where 0 <= i < j < N
    costs = [[0] * N for _ in range(N)]
    for i in range(N - 1):
        for j in range(i + 1, N):
            costs[i][j] = costs[j][i] = int(input_data[ptr])
            ptr += 1

    # The goal is to find a permutation P of {0, ..., N-1} that minimizes:
    # Sum_{i < j} cost(i, j) if (G(i, j) != H(P_i, P_j))
    
    # Generate all permutations of (0, ..., N-1)
    # P[i] is the vertex in H that corresponds to vertex i in G
    all_perms = permutations(range(N))
    
    # Precompute the list of all pairs (i, j) with i < j
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Calculate the cost for each permutation
    # We use a generator expression inside min() to avoid explicit for/while loops
    # The cost for a pair (i, j) is costs[P[i]][P[j]] if the edge status differs
    
    ans = min(
        sum(
            costs[p[i]][p[j]] for i, j in pairs if adj_G[i][j] != adj_H[p[i]][p[j]]
        )
        for p in all_perms
    )
    
    print(ans)

if __name__ == "__main__":
    solve()