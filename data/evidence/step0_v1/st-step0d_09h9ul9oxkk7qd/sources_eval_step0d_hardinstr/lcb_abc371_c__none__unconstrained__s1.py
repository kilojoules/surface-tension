import itertools
import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use a pointer-like approach via indexing to parse the input
    # Since we can't use while loops, we rely on the known structure of the input
    N = int(input_data[0])
    
    # Parse Graph G
    MG = int(input_data[1])
    # G_edges is a set of frozen sets for fast lookup
    # We use a list comprehension to extract the edges
    G_edges_list = [
        frozenset([int(input_data[2 + 2*i]), int(input_data[3 + 2*i])])
        for i in range(MG)
    ]
    G_adj = set(G_edges_list)
    
    # Parse Graph H
    # The starting index for MH is 2 + 2*MG
    MH_idx = 2 + 2*MG
    MH = int(input_data[MH_idx])
    H_edges_list = [
        frozenset([int(input_data[MH_idx + 1 + 2*i]), int(input_data[MH_idx + 2 + 2*i])])
        for i in range(MH)
    ]
    H_adj = set(H_edges_list)
    
    # Parse Cost Matrix A
    # The starting index for A is MH_idx + 1 + 2*MH
    A_start_idx = MH_idx + 1 + 2*MH
    # A is provided as a flattened list of A_{i,j} for 1 <= i < j <= N
    A_flat = [int(x) for x in input_data[A_start_idx:]]
    
    # To easily access A_{i,j}, we create a mapping from (i, j) to cost
    # There are N*(N-1)//2 pairs. We generate the pairs in the order they appear in input.
    pairs = [
        (i, j) 
        for i in range(1, N + 1) 
        for j in range(i + 1, N + 1)
    ]
    cost_map = dict(zip(pairs, A_flat))
    
    # We need to find a permutation P of (1...N) that minimizes the cost
    # Cost for a permutation P:
    # For every pair (i, j) with 1 <= i < j <= N:
    # If (edge (i, j) in G) != (edge (P_i, P_j) in H), add cost A_{P_i, P_j}
    # Note: The problem says we modify H to be isomorphic to G.
    # If we map vertex i of G to vertex P_i of H, then edge (i, j) in G 
    # must correspond to edge (P_i, P_j) in H.
    
    # Pre-calculate all possible pairs (i, j) for the cost summation
    all_pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    
    # Generate all permutations of 1...N
    perms = itertools.permutations(range(1, N + 1))
    
    # For each permutation, calculate the total cost
    # We use a nested list comprehension: 
    # Outer: iterate permutations
    # Inner: sum costs for all pairs (i, j)
    # We use a helper function to ensure we access cost_map with sorted keys (u, v) where u < v
    
    def get_cost(p):
        # p is the permutation where p[i-1] is the image of vertex i
        # For each pair (i, j) in G, check if edge exists in G and if edge (p[i-1], p[j-1]) exists in H
        # The cost is A_{p[i-1], p[j-1]} if they differ.
        return sum([
            cost_map[tuple(sorted((p[i-1], p[j-1])))]
            for i, j in all_pairs
            if (frozenset([i, j]) in G_adj) != (frozenset([p[i-1], p[j-1]]) in H_adj)
        ])

    # Calculate minimum cost across all permutations
    # Using map() to avoid explicit for-loop
    ans = min(map(get_cost, perms))
    print(ans)

if __name__ == "__main__":
    solve()