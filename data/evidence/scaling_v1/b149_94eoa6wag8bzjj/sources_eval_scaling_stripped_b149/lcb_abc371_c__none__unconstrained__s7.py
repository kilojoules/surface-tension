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
    
    # Graph G edges
    MG = int(input_data[ptr])
    ptr += 1
    G_edges_list = []
    for _ in range(MG):
        u = int(input_data[ptr])
        v = int(input_data[ptr+1])
        G_edges_list.append((u, v))
        ptr += 2
        
    # Graph H edges
    MH = int(input_data[ptr])
    ptr += 1
    H_edges_list = []
    for _ in range(MH):
        u = int(input_data[ptr])
        v = int(input_data[ptr+1])
        H_edges_list.append((u, v))
        ptr += 2
        
    # Cost matrix A
    # A is provided as a flattened triangle: A[1,2], A[1,3]... A[1,N], A[2,3]...
    # We map this to a 2D array indexed by (i, j) where i < j
    A_flat = list(map(int, input_data[ptr:]))
    
    # Pre-calculate the cost matrix A[i][j] for 1 <= i < j <= N
    # Using a dictionary for easy lookup: cost_map[(i, j)]
    # The input order is A_{1,2}, A_{1,3}... A_{1,N}, A_{2,3}... A_{2,N}, etc.
    cost_map = {}
    idx = 0
    for i in range(1, N):
        for j in range(i + 1, N + 1):
            cost_map[(i, j)] = A_flat[idx]
            idx += 1

    # Adjacency matrices for G and H
    # adj_G[i][j] = 1 if edge exists, else 0
    adj_G = [[0] * (N + 1) for _ in range(N + 1)]
    for u, v in G_edges_list:
        adj_G[u][v] = adj_G[v][u] = 1
        
    adj_H = [[0] * (N + 1) for _ in range(N + 1)]
    for u, v in H_edges_list:
        adj_H[u][v] = adj_H[v][u] = 1

    # We need to find a permutation P of (1...N) that minimizes:
    # Sum_{1 <= i < j <= N} cost(i, j) if (edge(i, j) in G != edge(P_i, P_j) in H)
    # Note: The problem says we pay A_{i,j} to flip edge (i, j) in H.
    # To make H isomorphic to G under permutation P, vertex i in G maps to P_i in H.
    # The edge (i, j) in G must correspond to edge (P_i, P_j) in H.
    # If they differ, we pay A_{P_i, P_j}.
    
    # Generate all permutations of 1...N
    perms = itertools.permutations(range(1, N + 1))
    
    # For each permutation P, calculate the total cost
    # Cost = sum_{1 <= i < j <= N} [ (adj_G[i][j] != adj_H[P[i-1]][P[j-1]]) * A[P_i, P_j] ]
    # We use a generator expression inside min()
    
    # To avoid loops in the cost calculation, we pre-generate all pairs (i, j)
    pairs = [(i, j) for i in range(1, N) for j in range(i + 1, N)]
    
    ans = min(
        sum(
            cost_map[tuple(sorted((p[i-1], p[j-1])))] 
            for i, j in pairs 
            if adj_G[i][j] != adj_H[p[i-1]][p[j-1]]
        )
        for p in perms
    )
    
    print(ans)

if __name__ == "__main__":
    solve()