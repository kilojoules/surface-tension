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
    # A is given as A_{1,2}, A_{1,3}... A_{1,N}, A_{2,3}...
    # We store it in a dictionary for easy access: (i, j) where i < j
    A_values = input_data[ptr:]
    
    # To map the flat A_values list to (i, j) pairs:
    # There are N*(N-1)//2 pairs.
    # We can generate the pairs in the order they are provided in the input.
    pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    cost_map = {pairs[k]: int(A_values[k]) for k in range(len(pairs))}

    # Adjacency matrices for G and H (using 0-indexing internally)
    # adj_G[i][j] == 1 if edge exists
    adj_G = [[0]*N for _ in range(N)]
    for u, v in G_edges_list:
        adj_G[u-1][v-1] = adj_G[v-1][u-1] = 1
        
    adj_H = [[0]*N for _ in range(N)]
    for u, v in H_edges_list:
        adj_H[u-1][v-1] = adj_H[v-1][u-1] = 1

    # We need to find a permutation P of (0, ..., N-1) that minimizes:
    # Sum_{i < j} cost(i, j) if (G has edge i,j != H has edge P[i], P[j])
    # Note: The cost A_{i,j} is associated with the vertices of H.
    # If we map vertex i of G to vertex P[i] of H, the cost to flip edge 
    # (P[i], P[j]) in H to match G is A_{min(P[i], P[j]), max(P[i], P[j])}.
    
    # Pre-calculate all possible pairs (i, j) for the summation
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # The cost function for a permutation P:
    # For every pair (i, j) in G, we check if its existence matches (P[i], P[j]) in H.
    # If they differ, we pay the cost of flipping edge (P[i], P[j]) in H.
    
    # To avoid loops, we use a generator expression inside min()
    # P is a permutation of 0...N-1
    # The cost is A_{P[i]+1, P[j]+1} if adj_G[i][j] != adj_H[P[i]][P[j]]
    
    ans = min(
        sum(
            cost_map[(min(p[i], p[j]) + 1, max(p[i], p[j]) + 1)]
            for i, j in all_pairs
            if adj_G[i][j] != adj_H[p[i]][p[j]]
        )
        for p in permutations(range(N))
    )
    
    print(ans)

if __name__ == "__main__":
    solve()