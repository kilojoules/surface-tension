import sys
from itertools import permutations

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to consume input linearly
    gen = input_data
    
    N = next(gen)
    
    # Graph G edges
    MG = next(gen)
    # We use a set of frozensets for efficient edge lookup
    # Each edge is represented as a frozenset({u, v})
    G_edges = {frozenset([next(gen), next(gen)]) for _ in range(MG)}
    
    # Graph H edges
    MH = next(gen)
    H_edges = {frozenset([next(gen), next(gen)]) for _ in range(MH)}
    
    # Cost matrix A_{i,j}
    # The input gives A_{i,j} for 1 <= i < j <= N
    # We store them in a dictionary with keys (i, j) where i < j
    costs = {}
    for i in range(1, N):
        for j in range(i + 1, N + 1):
            costs[(i, j)] = next(gen)
            
    # To check isomorphism, we try all permutations P of {1, ..., N}
    # G and H are isomorphic if there exists P such that:
    # (i, j) is an edge in G <=> (P_i, P_j) is an edge in H
    # The cost to transform H to be isomorphic to G under permutation P is:
    # Sum of A_{P_i, P_j} for all pairs (i, j) where the edge status differs.
    
    # Pre-calculate all possible pairs (i, j) with 1 <= i < j <= N
    all_pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    
    # Function to calculate cost for a specific permutation P
    # P is a tuple where P[i-1] is the image of vertex i
    def get_cost(P):
        total = 0
        for i, j in all_pairs:
            # Vertices in G are i and j. Their images in H are P[i-1] and P[j-1].
            # We need to check if the edge existence matches.
            # Edge in G:
            has_g = frozenset([i, j]) in G_edges
            # Edge in H:
            u, v = P[i-1], P[j-1]
            # Ensure u < v for dictionary lookup
            pair_h = (u, v) if u < v else (v, u)
            has_h = frozenset([u, v]) in H_edges
            
            if has_g != has_h:
                total += costs[pair_h]
        return total

    # Try all permutations of (1, ..., N)
    # map(get_cost, permutations(range(1, N + 1))) creates an iterator of costs
    # min() finds the minimum value among all permutations
    ans = min(map(get_cost, permutations(range(1, N + 1))))
    
    print(ans)

if __name__ == "__main__":
    solve()