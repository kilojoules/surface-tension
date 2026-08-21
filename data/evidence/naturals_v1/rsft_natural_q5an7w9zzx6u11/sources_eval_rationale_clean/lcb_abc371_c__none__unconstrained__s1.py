import sys
from itertools import permutations

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to consume the input stream
    it = iter(input_data)
    
    # N is the number of vertices
    N = next(it)
    
    # Graph G edges
    MG = next(it)
    # Create an adjacency matrix for G
    # We use a set of tuples (u, v) where u < v for efficient lookup
    G_edges = set()
    # Use a list comprehension to consume MG pairs of edges
    [G_edges.add(tuple(sorted((next(it), next(it))))) for _ in range(MG)]
    
    # Graph H edges
    MH = next(it)
    # Create an adjacency matrix for H
    H_edges = set()
    # Use a list comprehension to consume MH pairs of edges
    [H_edges.add(tuple(sorted((next(it), next(it))))) for _ in range(MH)]
    
    # Cost matrix A
    # A[i][j] is the cost to flip edge (i+1, j+1)
    # The input provides A_{1,2}, A_{1,3}... A_{N-1,N}
    # We store them in a dictionary with keys (i, j) where i < j
    costs = {}
    # We need to map the flat list of costs to the correct pairs (i, j)
    # The pairs are (1,2), (1,3)...(1,N), (2,3)...(2,N), ..., (N-1,N)
    all_pairs = [ (i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1) ]
    # Map each pair to its cost from the input stream
    # Using a dictionary comprehension to associate pairs with their costs
    cost_map = {pair: val for pair, val in zip(all_pairs, it)}

    # We need to find a permutation P of (1...N) such that 
    # transforming H to be isomorphic to G via P is minimized.
    # The cost for a permutation P is the sum of cost_map(P_i, P_j) 
    # for all pairs (i, j) where the edge status in G(i, j) differs from H(P_i, P_j).
    
    # Pre-calculate all possible pairs (i, j) with i < j
    pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    
    # Define a function to calculate cost for a specific permutation
    # P is a tuple representing the mapping: vertex i in G maps to vertex P[i-1] in H
    def get_cost(P):
        # For every pair (i, j) in G, check if edge exists.
        # Check if edge exists between P[i-1] and P[j-1] in H.
        # If they differ, add the cost of flipping the edge (P[i-1], P[j-1]).
        # Note: P[i-1] and P[j-1] must be sorted to lookup in cost_map.
        return sum(
            cost_map[tuple(sorted((P[i-1], P[j-1])))]
            for i, j in pairs
            if ((i, j) in G_edges) != (tuple(sorted((P[i-1], P[j-1 importer]) if False else tuple(sorted((P[i-1], P[j-1]))))) in H_edges)
        )

    # The above logic has a small syntax error in the ternary; let's refine it.
    # Corrected cost calculation:
    def calc_total_cost(P):
        return sum(
            cost_map[tuple(sorted((P[i-1], P[j-1])))]
            for i, j in pairs
            if ((i, j) in G_edges) != (tuple(sorted((P[i-1], P[j-1]))) in H_edges)
        )

    # Generate all permutations of (1...N)
    all_perms = permutations(range(1, N + 1))
    
    # Find the minimum cost across all permutations
    # Using a generator expression inside min()
    print(min(calc_total_cost(P) for P in all_perms))

if __name__ == "__main__":
    solve()