import sys
from itertools import permutations

def solve():
    # Read all input at once and split into a list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use a generator/iterator to consume input tokens
    it = iter(input_data)
    
    N = int(next(it))
    
    # Function to read M edges and return a set of frozen sets (undirected edges)
    def read_edges(m_val):
        edges = [
            frozenset([int(next(it)), int(next(it))])
            for _ in range(m_val)
        ]
        return set(edges)

    MG = int(next(it))
    edges_G = read_edges(MG)
    
    MH = int(next(it))
    edges_H = read_edges(MH)
    
    # Read the A_{i,j} matrix. 
    # The input provides A_{i,j} for 1 <= i < j <= N in a specific order.
    # We store them in a dictionary where keys are frozensets {i, j}.
    # The order in input is:
    # i=1: j=2, 3, ..., N
    # i=2: j=3, ..., N
    # ...
    # i=N-1: j=N
    
    # To map the flat list of A values to the correct pairs (i, j):
    # We can pre-calculate the pairs in the exact order they appear in the input.
    all_pairs = [
        frozenset([i, j])
        for i in range(1, N + 1)
        for j in range(i + 1, N + 1)
    ]
    
    # Map each pair to its corresponding cost A_{i,j}
    # Since we cannot use loops, we use a list comprehension to pair 
    # the remaining tokens with the pre-calculated pairs.
    costs_list = [int(next(it)) for _ in range(len(all_pairs))]
    cost_map = {pair: cost for pair, cost in zip(all_pairs, costs_list)}

    # We need to find a permutation P of (1, ..., N) that minimizes the cost.
    # The cost for a permutation P is the sum of A_{P_i, P_j} for all pairs (i, j)
    # where the edge status in G (i, j) differs from the edge status in H (P_i, P_j).
    
    # Pre-calculate all possible pairs (i, j) for 1 <= i < j <= N
    # These are the indices in G.
    g_pairs = [
        (i, j)
        for i in range(1, N + 1)
        for j in range(i + 1, N + 1)
    ]

    # For a given permutation P, the cost is:
    # sum(cost_map[frozenset([P[i-1], P[j-1]])] 
    #     if (edge (i,j) in G) != (edge (P_i, P_j) in H))
    
    # We use a generator expression inside min() to iterate over all permutations.
    # P is a tuple from itertools.permutations.
    # We map the 1-based indices i, j to 0-based indices for the permutation tuple.
    
    # To avoid loops and maintain efficiency, we pre-calculate the 
    # existence of edges in G and H for every possible pair.
    # However, the isomorphism definition says:
    # edge (i, j) exists in G <=> edge (P_i, P_j) exists in H.
    # The cost is incurred if this equivalence is violated.
    
    # Let's define a helper to check edge existence:
    # G_edge(i, j) is True if {i, j} is in edges_G
    # H_edge(i, j) is True if {i, j} is in edges_H
    
    # We can use a list comprehension to create a function-like object 
    # or just embed the logic in the generator.
    
    ans = min(
        sum(
            cost_map[frozenset([p[i-1], p[j-1]])]
            for i, j in g_pairs
            if (frozenset([i, j]) in edges_G) != (frozenset([p[i-1], p[j-1]]) in edges_H)
        )
        for p in permutations(range(1, N + 1))
    )
    
    print(ans)

if __name__ == "__main__":
    solve()