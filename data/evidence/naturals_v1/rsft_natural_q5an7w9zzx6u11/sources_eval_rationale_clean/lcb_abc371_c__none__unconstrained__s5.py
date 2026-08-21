import sys
from itertools import permutations

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use a pointer-like approach via indexing to parse the input
    # since we cannot use while/for loops to iterate through the input stream.
    N = int(input_data[0])
    
    # Parse M_G and edges of G
    M_G = int(input_data[1])
    # G_edges is a set of frozen sets for fast lookup
    # We use a list comprehension to extract the edges
    G_edges_list = [
        frozenset([int(input_data[2 + 2*i]), int(input_data[3 + 2*i])])
        for i in range(M_G)
    ]
    G_adj = set(G_edges_list)
    
    # Parse M_H and edges of H
    # Offset for M_H is 2 + 2*M_G
    offset_MH = 2 + 2 * M_G
    M_H = int(input_data[offset_MH])
    H_edges_list = [
        frozenset([int(input_data[offset_MH + 1 + 2*i]), int(input_data[offset_MH + 2 + 2*i])])
        for i in range(M_H)
    ]
    H_adj = set(H_edges_list)
    
    # Parse A_{i,j} matrix
    # The A values start after the H edges
    offset_A = offset_MH + 1 + 2 * M_H
    # A is provided as a flattened list of A_{i,j} for 1 <= i < j <= N
    # We map them to a dictionary with keys (i, j) where i < j
    # To avoid loops, we generate the indices (i, j) first
    all_pairs = [ (i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1) ]
    A_vals = input_data[offset_A:]
    A_map = dict(zip(all_pairs, map(int, A_vals)))

    # The core logic:
    # For a permutation P, vertex i in G maps to P[i-1] in H.
    # An edge (i, j) in G must exist in H between P[i-1] and P[j-1].
    # If the edge status differs, we pay A_{min(P[i-1], P[j-1]), max(P[i-1], P[j-1])}.
    
    # We generate all permutations of (1, ..., N)
    # For each permutation, we calculate the total cost across all pairs (i, j)
    # The result is the minimum of these total costs.
    
    # To avoid loops, we use nested comprehensions:
    # 1. Iterate over all permutations P
    # 2. For each P, sum the costs of all pairs (i, j)
    # 3. Find the minimum sum
    
    # Pre-calculate all pairs (i, j) for G to avoid redundant work in the inner loop
    g_pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    
    ans = min(
        sum(
            A_map[(min(p[i-1], p[j-1]), max(p[i-1], p[j-1]))]
            if (frozenset([i, j]) in G_adj) != (frozenset([p[i-1], p[j-1]]) in H_adj)
            else 0
            for i, j in g_pairs
        )
        for p in permutations(range(1, N + 1))
    )
    
    print(ans)

if __name__ == "__main__":
    solve()