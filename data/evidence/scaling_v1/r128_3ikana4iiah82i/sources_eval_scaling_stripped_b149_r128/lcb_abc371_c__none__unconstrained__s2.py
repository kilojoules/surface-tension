import sys
from itertools import permutations

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    data = map(int, input_data)
    
    # Use a generator/iterator to consume input sequentially
    it = data
    
    # N: Number of vertices
    N = next(it)
    
    # M_G: Number of edges in G
    M_G = next(it)
    # Build adjacency matrix for G
    # We use a set of frozen sets for edges to allow O(1) lookup
    # Each edge is a frozenset({u, v})
    edges_G = {frozenset([next(it), next(it)]) for _ in range(M_G)}
    
    # M_H: Number of edges in H
    M_H = next(it)
    # Build adjacency matrix for H
    edges_H = {frozenset([next(it), next(it)]) for _ in range(M_H)}
    
    # A: Costs for toggling edges (i, j)
    # The input provides A_{i,j} for 1 <= i < j <= N
    # We store them in a dictionary keyed by frozenset({i, j})
    # We generate the keys based on the order they appear in the input
    # The order is (1,2), (1,3)...(1,N), (2,3)...(2,N), etc.
    cost_map = {
        frozenset([i, j]): next(it)
        for i in range(1, N + 1)
        for j in range(i + 1, N + 1)
    }

    # To make G and H isomorphic, we need a permutation P of {1...N}
    # such that for all i < j, edge (i, j) is in G iff edge (P_i, P_j) is in H.
    # If this condition is not met, we must pay the cost to toggle the edge in H.
    # The cost for a permutation P is the sum of cost_map({P_i, P_j}) 
    # for all i < j where (edge (i, j) in G) != (edge (P_i, P_j) in H).
    
    # Generate all possible permutations of vertices 1...N
    all_permutations = permutations(range(1, N + 1))
    
    # For each permutation, calculate the total cost to make H isomorphic to G
    # We use a generator expression inside min() to find the minimum cost.
    # We iterate through all pairs (i, j) with i < j.
    
    # Pre-calculate all possible pairs (i, j) to avoid loops inside the generator
    all_pairs = [frozenset([i, j]) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    
    # The core logic:
    # For a permutation P, the vertex i in G maps to P[i-1] in H.
    # The edge (i, j) in G maps to the edge (P[i-1], P[j-1]) in H.
    # We pay if (edge (i, j) exists in G) XOR (edge (P[i-1], P[j-1]) exists in H).
    
    ans = min(
        sum(
            cost_map[frozenset([p[i], p[j]])]
            for i, j in ( (a, b) for a in range(N) for b in range(a + 1, N) )
            if (frozenset([i + 1, j + 1]) in edges_G) != (frozenset([p[i], p[j]]) in edges_H)
        )
        for p in all_permutations
    )
    
    print(ans)

if __name__ == "__main__":
    solve()