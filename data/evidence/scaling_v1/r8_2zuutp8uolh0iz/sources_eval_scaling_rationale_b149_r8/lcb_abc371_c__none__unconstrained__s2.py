import sys
from itertools import permutations

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use a pointer-like approach with a list to simulate reading
    # Since we can't use while loops, we map the input to variables
    N = int(input_data[0])
    
    # Extract M_G and edges of G
    M_G = int(input_data[1])
    g_edges_raw = input_data[2 : 2 + 2 * M_G]
    
    # Extract M_H and edges of H
    current_pos = 2 + 2 * M_G
    M_H = int(input_data[current_pos])
    h_edges_raw = input_data[current_pos + 1 : current_pos + 1 + 2 * M_H]
    
    # Extract A matrix
    a_flat = list(map(int, input_data[current_pos + 1 + 2 * M_H :]))
    
    # Build adjacency matrices for G and H
    # G_adj[i][j] == 1 if edge exists
    G_adj = [[0] * N for _ in range(N)]
    # Using a helper to fill G_adj without loops is tricky, 
    # but we can use a list comprehension to build the matrix based on edge existence.
    g_edges = [(int(g_edges_raw[i]), int(g_edges_raw[i+1])) for i in range(0, 2 * M_G, 2)]
    G_matrix = [[1 if (i+1, j+1) in g_edges or (j+1, i+1) in g_edges else 0 
                 for j in range(N)] for i in range(N)]
    
    h_edges = [(int(h_edges_raw[i]), int(h_edges_raw[i+1])) for i in range(0, 2 * M_H, 2)]
    H_matrix = [[1 if (i+1, j+1) in h_edges or (j+1, i+1) in h_edges else 0 
                 for j in range(N)] for i in range(N)]
    
    # Reconstruct A matrix from the flat list
    # A[i][j] is the cost to flip edge (i, j)
    # The input gives A_{1,2}, A_{1,3}... A_{N-1,N}
    # We create a lookup dictionary for costs A_{i,j} where i < j
    # The number of A values is N(N-1)//2
    # We can map the flat list to pairs (i, j) using a comprehension
    cost_pairs = [ (i, j) for i in range(1, N+1) for j in range(i+1, N+1) ]
    A_dict = dict(zip(cost_pairs, a_flat))
    
    # We need to find a permutation P of (0, ..., N-1) such that 
    # the cost to make H isomorphic to G is minimized.
    # Cost for permutation P: 
    # Sum over i < j of A_{P[i], P[j]} if G_matrix[i][j] != H_matrix[P[i]][P[j]]
    # Wait, the definition of isomorphism is: 
    # edge (i, j) in G <=> edge (P[i], P[j]) in H.
    # The cost is paid to change H. So if G(i, j) != H(P[i], P[j]), 
    # we pay A_{min(P[i], P[j]), max(P[i], P[j])}.
    
    # Generate all permutations of 0...N-1 (representing the mapping)
    perms = permutations(range(N))
    
    # For a fixed permutation P, calculate total cost
    # We iterate over all pairs (i, j) with 0 <= i < j < N
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # The cost for a permutation P is the sum of A_{P[i], P[j]} 
    # for all pairs (i, j) where G_matrix[i][j] != H_matrix[P[i]][P[j]]
    # Note: P[i] and P[j] are 0-indexed, but A_dict uses 1-indexed.
    
    # To avoid loops, we use a nested comprehension:
    # min( [sum([cost for pair in pairs if ...]) for P in perms] )
    
    # Since we need to map P[i] to 1-indexed for A_dict:
    # cost = A_dict[(min(P[i]+1, P[j]+1), max(P[i]+1, P[j]+1))]
    
    ans = min([
        sum([
            A_dict[(min(p[i]+1, p[j]+1), max(p[i]+1, p[j]+1))]
            for i, j in pairs 
            if G_matrix[i][j] != H_matrix[p[i]][p[j]]
        ])
        for p in perms
    ])
    
    print(ans)

if __name__ == "__main__":
    solve()