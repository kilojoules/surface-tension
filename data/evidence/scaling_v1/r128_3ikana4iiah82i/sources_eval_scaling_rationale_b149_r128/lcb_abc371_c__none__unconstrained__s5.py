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
    g_edges = []
    for _ in range(MG):
        g_edges.append((int(input_data[ptr]), int(input_data[ptr+1])))
        ptr += 2
        
    # Graph H edges
    MH = int(input_data[ptr])
    ptr += 1
    h_edges = []
    for _ in range(MH):
        h_edges.append((int(input_data[ptr]), int(input_data[ptr+1])))
        ptr += 2
        
    # Cost matrix A
    # A[i][j] is the cost to flip edge (i+1, j+1)
    # The input provides A_{1,2}, A_{1,3}... A_{N-1,N}
    # We store it in a dictionary for easy access: (i, j) -> cost
    costs_list = input_data[ptr:]
    
    # To handle the cost matrix without loops, we map the flat list to pairs
    # There are N*(N-1)//2 costs.
    # We can use a dictionary comprehension to store costs for pairs (i, j) where i < j.
    # We generate the pairs (i, j) first.
    pairs = [ (i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1) ]
    cost_map = { pair: int(cost) for pair, cost in zip(pairs, costs_list) }

    # Adjacency matrices for G and H
    # G_adj[i][j] = 1 if edge exists, else 0
    G_adj = [[0] * (N + 1) for _ in range(N + 1)]
    for u, v in g_edges:
        G_adj[u][v] = G_adj[v][u] = 1
        
    H_adj = [[0] * (N + 1) for _ in range(N + 1)]
    for u, v in h_edges:
        H_adj[u][v] = H_adj[v][u] = 1

    # We need to find a permutation P of (1...N) that minimizes the cost.
    # The cost for a permutation P is the sum of cost_map(P_i, P_j) 
    # for all i < j where G_adj[i][j] != H_adj[P_i][P_j].
    
    # Generate all permutations of 1...N
    all_perms = permutations(range(1, N + 1))
    
    # For each permutation, calculate the total cost.
    # We use a generator expression inside min().
    # Note: P is a tuple, so P[i-1] is the vertex mapped to i.
    
    ans = min(
        sum(
            cost_map[tuple(sorted((P[i-1], P[j-1])))]
            for i in range(1, N + 1)
            for j in range(i + 1, N + 1)
            if G_adj[i][j] != H_adj[P[i-1]][P[j-1]]
        )
        for P in all_perms
    )
    
    print(ans)

if __name__ == "__main__":
    solve()