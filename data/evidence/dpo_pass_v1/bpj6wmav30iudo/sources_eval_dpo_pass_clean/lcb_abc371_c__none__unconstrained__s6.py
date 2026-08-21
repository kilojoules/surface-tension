import sys
from itertools import permutations

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use an iterator to consume the input stream sequentially
    it = iter(input_data)
    
    # Extract N and M_G
    N = next(it)
    M_G = next(it)
    
    # Build adjacency matrix for G
    # We use a set of frozen sets to represent edges for O(1) lookup
    G_edges = {frozenset([next(it), next(it)]) for _ in range(M_G)}
    
    # Extract M_H
    M_H = next(it)
    
    # Build adjacency matrix for H
    H_edges = {frozenset([next(it), next(it)]) for _ in range(M_H)}
    
    # Read the cost matrix A
    # A[i][j] is the cost to toggle edge (i+1, j+1)
    # We store it as a dictionary where keys are frozensets of vertices
    # to match the edge representation.
    # The input for A is given as A_{1,2}, A_{1,3}... A_{N-1,N}
    # We generate the pairs (i, j) corresponding to the input order.
    cost_pairs = [frozenset([i, j]) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    A_values = [next(it) for _ in range(len(cost_pairs))]
    A = dict(zip(cost_pairs, A_values))

    # A permutation P maps vertex i in G to vertex P[i-1] in H.
    # For a fixed P, the cost is the sum of A_{P[i-1], P[j-1]} for all pairs (i, j)
    # where the edge status in G (i, j) differs from the edge status in H (P[i-1], P[j-1]).
    
    # Generate all permutations of (1, ..., N)
    all_perms = permutations(range(1, N + 1))
    
    # For each permutation, calculate the total cost
    # We iterate over all possible edges (i, j) in a graph of size N
    possible_edges = [frozenset([i, j]) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    
    # The cost for a specific permutation P is:
    # Sum_{i < j} A_{P[i-1], P[j-1]} if (edge(i, j) in G) != (edge(P[i-1], P[j-1]) in H)
    
    # To avoid loops, we use a nested list comprehension/map 
    # and the min() function.
    ans = min(
        sum(
            A[frozenset([p[i-1], p[j-1]])]
            for i in range(1, N + 1)
            for j in range(i + 1, N + 1)
            if (frozenset([i, j]) in G_edges) != (frozenset([p[i-1], p[j-1]]) in H_edges)
        )
        for p in all_perms
    )
    
    print(ans)

if __name__ == "__main__":
    solve()