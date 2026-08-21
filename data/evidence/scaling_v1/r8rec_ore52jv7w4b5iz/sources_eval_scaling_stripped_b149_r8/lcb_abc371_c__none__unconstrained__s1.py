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
        G_edges.append((int(input_data[ptr]), int(input_data[ptr+1])))
        ptr += 2
        
    # Graph H edges
    MH = int(input_data[ptr])
    ptr += 1
    H_edges = []
    for _ in range(MH):
        H_edges.append((int(input_data[ptr]), int(input_data[ptr+1])))
        ptr += 2
        
    # Cost matrix A
    # The input provides A_{i,j} for 1 <= i < j <= N
    # We store them in a dictionary for easy access: (i, j) where i < j
    A_vals = input_data[ptr:]
    
    # Map (i, j) to the cost A_{i,j}
    # There are N*(N-1)//2 such pairs.
    # We can generate the pairs (i, j) in the order they appear in the input.
    all_pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    cost_map = {pair: int(val) for pair, val in zip(all_pairs, A_vals)}

    # Adjacency matrices for G and H (using 0-indexing internally)
    # adj_G[i][j] == 1 if edge exists between vertex i+1 and j+1
    adj_G = [[0]*N for _ in range(N)]
    for u, v in G_edges:
        adj_G[u-1][v-1] = adj_G[v-1][u-1] = 1
        
    adj_H = [[0]*N for _ in range(N)]
    for u, v in H_edges:
        adj_H[u-1][v-1] = adj_H[v-1][u-1] = 1

    # We need to find a permutation P of (0, ..., N-1) that minimizes 
    # sum_{i < j} cost(i+1, j+1) if (adj_G[P[i]][P[j]] != adj_H[i][j])
    # Note: The problem says we modify H to be isomorphic to G.
    # This means there exists P such that edge (i, j) in G <=> edge (P[i], P[j]) in H.
    # The cost is paid to change edges in H.
    # So for a fixed P, for every pair (i, j) with i < j:
    # If adj_G[i][j] != adj_H[P[i]][P[j]], we pay A_{P[i]+1, P[j]+1}.
    
    # To avoid loops, we use a generator expression inside min()
    # We iterate through all permutations of range(N)
    ans = min(
        sum(
            cost_map[tuple(sorted((p[i] + 1, p[j] + 1)))]
            for i in range(N)
            for j in range(i + 1, N)
            if adj_G[i][j] != adj_H[p[i]][p[j]]
        )
        for p in permutations(range(N))
    )
    
    print(ans)

if __name__ == "__main__":
    solve()