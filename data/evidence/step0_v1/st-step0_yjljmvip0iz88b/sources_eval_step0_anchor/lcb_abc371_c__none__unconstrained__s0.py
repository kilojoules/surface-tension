import sys
from itertools import permutations

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to consume the input stream
    it = iter(input_data)
    
    N = next(it)
    
    # Graph G edges
    Mg = next(it)
    g_edges = [tuple(sorted((next(it), next(it)))) for _ in range(Mg)]
    
    # Graph H edges
    Mh = next(it)
    h_edges = [tuple(sorted((next(it), next(it)))) for _ in range(Mh)]
    
    # Cost matrix A
    # A[i][j] will store the cost to flip edge (i+1, j+1)
    # We read the costs in the order provided: (1,2), (1,3)...(1,N), (2,3)...(N-1,N)
    # We can store these in a dictionary with sorted tuple keys for easy lookup
    all_pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    costs = {pair: val for pair, val in zip(all_pairs, it)}

    # Adjacency matrices for G and H for O(1) lookup
    # Using sets of sorted tuples for G and H
    set_g = set(g_edges)
    set_h = set(h_edges)

    # We need to find a permutation P of (1...N) such that 
    # transforming H to match G under P is minimized.
    # The cost for a permutation P is the sum of costs A_{P_i, P_j} 
    # for all pairs (i, j) where (edge in G) != (edge in H after permutation).
    # Specifically, for 1 <= i < j <= N:
    # If (i, j) is an edge in G, we need (P_i, P_j) to be an edge in H.
    # If (i, j) is NOT an edge in G, we need (P_i, P_j) to NOT be an edge in H.
    # If the condition is not met, we pay A_{P_i, P_j}.
    
    # Pre-calculate all pairs (i, j) with 1 <= i < j <= N
    pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    
    # Iterate through all permutations of (1...N)
    # For each permutation P, calculate the total cost
    # P is treated as a 1-indexed mapping: P[i-1] is the image of vertex i
    
    def calculate_cost(p):
        # p is a tuple (P1, P2, ..., PN)
        # For each pair (i, j) in G, check if edge exists.
        # Compare with edge (p[i-1], p[j-1]) in H.
        # If they differ, add cost A_{p[i-1], p[j-1]}
        return sum(
            costs[tuple(sorted((p[i-1], p[j-1]))))]
            for i, j in pairs
            if ((i, j) in set_g) != (tuple(sorted((p[i-1], p[j-1]))) in set_h)
        )

    # Use min() with a generator expression to find the minimum cost across all permutations
    print(min(calculate_cost(p) for p in permutations(range(1, N + 1))))

if __name__ == "__main__":
    solve()