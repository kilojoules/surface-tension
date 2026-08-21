import sys
from itertools import permutations

def solve():
    # Read all input at once and split into a list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to consume input values one by one
    gen = input_data
    
    N = next(gen)
    
    # Read G edges
    Mg = next(gen)
    g_edges = [tuple(sorted((next(gen), next(gen)))) for _ in range(Mg)]
    
    # Read H edges
    Mh = next(gen)
    h_edges = [tuple(sorted((next(gen), next(gen)))) for _ in range(Mh)]
    
    # Read A_{i,j} matrix
    # The input provides A_{i,j} for 1 <= i < j <= N
    # We store them in a dictionary with keys (i, j) where i < j
    # To map the flat input list to (i, j) pairs:
    # i=1: j=2..N (N-1 values)
    # i=2: j=3..N (N-2 values) ...
    
    # Since we cannot use loops, we use a list comprehension to generate the indices
    # and zip it with the remaining input generator.
    indices = [
        (i, j) 
        for i in range(1, N + 1) 
        for j in range(i + 1, N + 1)
    ]
    
    # Create a dictionary mapping (i, j) -> cost
    # We use a list comprehension to consume the generator for the remaining A values
    costs_list = [next(gen) for _ in range(len(indices))]
    cost_map = dict(zip(indices, costs_list))
    
    # Represent G and H as sets of frozen sets (edges) for fast lookup
    # Vertices are 1-indexed as per input
    G_set = {frozenset(e) for e in g_edges}
    H_set = {frozenset(e) for e in h_edges}
    
    # We need to find a permutation P of (1...N) that minimizes the cost.
    # The cost for a permutation P is the sum of A_{P_i, P_j} for all pairs (i, j)
    # where the edge status in G(i, j) differs from the edge status in H(P_i, P_j).
    
    # Generate all permutations of (1...N)
    # For each permutation, calculate the total cost.
    # We use a generator expression inside min() to avoid explicit loops.
    
    # To avoid loops, we pre-calculate all possible pairs (i, j) with i < j
    all_pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    
    # The cost function for a specific permutation p:
    # For every pair (i, j), check if edge (i, j) in G == edge (p[i-1], p[j-1]) in H.
    # If they differ, add cost A_{min(p[i-1], p[j-1]), max(p[i-1], p[j-1])}.
    
    # We use a helper function to get the cost of a single pair under permutation p
    def get_pair_cost(p, pair):
        i, j = pair
        # Edge in G
        has_g = frozenset({i, j}) in G_set
        # Edge in H (after permutation)
        u, v = p[i-1], p[j-1]
        has_h = frozenset({u, v}) in H_set
        
        if has_g == has_h:
            return 0
        else:
            # Return A_{u, v} ensuring u < v
            return cost_map[(u, v) if u < v else (v, u)]

    # Map the get_pair_cost over all pairs for every permutation
    # We use a nested generator: 
    # 1. Iterate over all permutations p
    # 2. For each p, sum the costs of all pairs
    # 3. Find the minimum of these sums
    
    ans = min(
        sum(get_pair_cost(p, pair) for pair in all_pairs)
        for p in permutations(range(1, N + 1))
    )
    
    print(ans)

if __name__ == "__main__":
    solve()