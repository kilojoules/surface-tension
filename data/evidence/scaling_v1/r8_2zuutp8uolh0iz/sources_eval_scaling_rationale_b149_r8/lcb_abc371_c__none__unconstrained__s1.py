import sys
from itertools import permutations

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use a pointer-like approach via a list to simulate reading from a stream
    # since we cannot use loops to iterate through the input.
    # However, since N is small, we can just map the flat list.
    
    N = int(input_data[0])
    
    # Extract M_G and the edges of G
    MG = int(input_data[1])
    g_edges_raw = input_data[2 : 2 + 2 * MG]
    
    # Extract M_H and the edges of H
    current_idx = 2 + 2 * MG
    MH = int(input_data[current_idx])
    h_edges_raw = input_data[current_idx + 1 : current_idx + 1 + 2 * MH]
    
    # Extract the cost matrix A
    # A is provided as a flattened upper triangle
    a_raw = input_data[current_idx + 1 + 2 * MH :]
    
    # Build adjacency matrices for G and H
    # G_adj[i][j] = 1 if edge exists, else 0
    # We use a dictionary or a 2D list. Since we can't use loops, 
    # we use set comprehensions for edge lookup.
    g_edges = {tuple(sorted((int(g_edges_raw[i]), int(g_edges_raw[i+1])))) 
               for i in range(0, 2 * MG, 2)}
    h_edges = {tuple(sorted((int(h_edges_raw[i]), int(h_edges_raw[i+1])))) 
               for i in range(0, 2 * MH, 2)}
    
    # Map the cost matrix A into a dictionary for easy access: A[(i, j)]
    # The input format for A is A_{1,2}, A_{1,3}... A_{1,N}, A_{2,3}...
    # We generate the indices (i, j) using a list comprehension.
    cost_indices = [ (i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1) ]
    cost_map = dict(zip(cost_indices, map(int, a_raw)))

    # For a given permutation P, the cost is the sum of A_{P_i, P_j} 
    # for all pairs (i, j) where the edge status in G(i, j) differs from H(P_i, P_j).
    # Note: The problem says we modify H to be isomorphic to G.
    # This means there exists P such that (i, j) is an edge in G iff (P_i, P_j) is an edge in H.
    # The cost is paid to flip edges in H.
    # If G has edge (i, j) and H does NOT have edge (P_i, P_j), we pay A_{P_i, P_j}.
    # If G does NOT have edge (i, j) and H HAS edge (P_i, P_j), we pay A_{P_i, P_j}.
    
    # We iterate over all permutations of (1, ..., N)
    # P maps vertex i of G to vertex P[i-1] of H.
    all_perms = permutations(range(1, N + 1))
    
    # For each permutation, calculate the total cost
    # We check all pairs 1 <= i < j <= N
    pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    
    def calculate_cost(p):
        # p is a tuple representing the permutation
        # For each pair (i, j) in G, check if edge status matches (p[i-1], p[j-1]) in H
        return sum(
            cost_map[tuple(sorted((p[i-1], p[j-1]))))]
            for i, j in pairs
            if ((i, j) in g_edges) != (tuple(sorted((p[i-1], p[j-1]))) in h_edges)
        )

    # Find the minimum cost across all permutations
    print(min(map(calculate_cost, all_perms)))

if __name__ == "__main__":
    solve()