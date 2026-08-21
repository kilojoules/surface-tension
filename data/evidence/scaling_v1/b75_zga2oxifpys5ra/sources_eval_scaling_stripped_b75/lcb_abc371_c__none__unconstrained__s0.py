import sys
from itertools import permutations

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use a generator/iterator to consume input
    it = iter(input_data)
    
    # N is the number of vertices
    N = int(next(it))
    
    # Read G edges
    MG = int(next(it))
    G_edges = [
        (int(next(it)), int(next(it))) 
        for _ in range(MG)
    ]
    
    # Read H edges
    MH = int(next(it))
    H_edges = [
        (int(next(it)), int(next(it))) 
        for _ in range(MH)
    ]
    
    # Read A_{i,j} costs
    # The costs are provided in a specific triangular format:
    # A_{1,2}, A_{1,3}... A_{1,N}
    # A_{2,3}... A_{2,N}
    # ...
    # A_{N-1,N}
    # We map these into a dictionary for O(1) access: {(i, j): cost} where i < j
    all_costs = [int(next(it)) for _ in range(N * (N - 1) // 2)]
    
    # To map the flat list to (i, j) pairs:
    # Row 1 has N-1 elements, Row 2 has N-2, etc.
    cost_map = {}
    curr = 0
    for i in range(1, N):
        for j in range(i + 1, N + 1):
            cost_map[(i, j)] = all_costs[curr]
            curr += 1

    # Adjacency matrices for G and H (using sets of frozensets for fast lookup)
    # G_adj: set of edges in G
    # H_adj: set of edges in H
    G_adj = {frozenset(edge) for edge in G_edges}
    H_adj = {frozenset(edge) for edge in H_edges}

    # We need to find a permutation P of (1...N) that minimizes the cost.
    # The cost for a permutation P is the sum of A_{P_i, P_j} for all pairs (i, j)
    # where the existence of edge (i, j) in G differs from the existence of edge (P_i, P_j) in H.
    
    # Pre-calculate all possible edges in a graph of size N
    all_pairs = [
        (i, j) 
        for i in range(1, N + 1) 
        for j in range(i + 1, N + 1)
    ]

    # We use a helper to get the cost of a specific permutation
    # To optimize, we use a list comprehension inside sum()
    # We use a closure or a function to avoid loops.
    
    def calculate_cost(p):
        # p is a permutation of (1...N)
        # For every pair (i, j) in G, check if edge (p[i-1], p[j-1]) in H matches
        # Note: p is 0-indexed, so vertex i is at p[i-1]
        return sum(
            cost_map[tuple(sorted((p[i-1], p[j-1])))]
            for i, j in all_pairs
            if (frozenset((i, j)) in G_adj) != (frozenset((p[i-1], p[j-1])) in H_adj)
        )

    # Try all N! permutations and find the minimum cost
    # N <= 8, so 8! = 40,320, which is well within limits for Python
    ans = min(calculate_cost(p) for p in permutations(range(1, N + 1)))
    
    print(ans)

if __name__ == "__main__":
    solve()