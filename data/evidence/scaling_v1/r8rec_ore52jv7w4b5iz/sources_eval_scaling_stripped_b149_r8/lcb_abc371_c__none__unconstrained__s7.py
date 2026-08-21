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
    # A[i][j] is the cost to toggle edge (i, j)
    # The input gives A_{1,2}, A_{1,3}... A_{N-1,N}
    # We build a 2D list for easy access.
    A_vals = input_data[ptr:]
    
    # To map A_{i,j} to a matrix, we can use a dictionary or a list of lists.
    # Since N is small (<= 8), we can pre-calculate the indices.
    # The number of A values is N*(N-1)//2.
    
    # Create a cost lookup table cost_map[(i, j)] = cost
    # We need to figure out which A_val corresponds to which (i, j)
    # The order is (1,2), (1,3)...(1,N), (2,3)...(2,N), ..., (N-1, N)
    pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    cost_map = {pairs[k]: int(A_vals[k]) for k in range(len(pairs))}

    # Adjacency matrices for G and H (using 0-indexing internally)
    adj_G = [[0] * N for _ in range(N)]
    for u, v in G_edges:
        adj_G[u-1][v-1] = adj_G[v-1][u-1] = 1
        
    adj_H = [[0] * N for _ in range(N)]
    for u, v in H_edges:
        adj_H[u-1][v-1] = adj_H[v-1][u-1] = 1

    # We want to find a permutation P of (0, ..., N-1) such that
    # the cost to make H isomorphic to G via P is minimized.
    # The cost for a permutation P is the sum of cost_map[(i+1, j+1)] 
    # for all i < j where (adj_G[P[i]][P[j]] != adj_H[i][j])
    # Wait, the definition of isomorphism says:
    # edge (i, j) in G <=> edge (P_i, P_j) in H.
    # Let's use the permutation P where vertex i in G is mapped to vertex P[i] in H.
    # Then edge (i, j) exists in G iff edge (P[i], P[j]) exists in H.
    # The cost is incurred if we need to change the edge status of (P[i], P[j]) in H
    # to match the status of (i, j) in G.
    
    # Pre-calculate all possible pairs (i, j) with i < j
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # We iterate through all permutations of range(N)
    # For each permutation P, we calculate the total cost.
    # Cost = sum(cost_map[(min(P[i], P[j])+1, max(P[i], P[j])+1)]) 
    #        for (i, j) in all_pairs if adj_G[i][j] != adj_H[P[i]][P[j]]
    
    # To optimize, we use a generator expression inside min()
    ans = min(
        sum(
            cost_map[(min(P[i], P[j]) + 1, max(P[i], P[j]) + 1)]
            for i, j in all_pairs
            if adj_G[i][j] != adj_H[P[i]][P[j]]
        )
        for P in permutations(range(N))
    )
    
    print(ans)

if __name__ == "__main__":
    solve()