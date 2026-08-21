import sys
from itertools import permutations

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract values one by one
    gen = input_data
    
    N = next(gen)
    
    # Graph G edges
    M_G = next(gen)
    G_edges = [tuple(sorted((next(gen), next(gen)))) for _ in range(M_G)]
    
    # Graph H edges
    M_H = next(gen)
    H_edges = [tuple(sorted((next(gen), next(gen)))) for _ in range(M_H)]
    
    # Cost matrix A_{i,j}
    # The input provides A_{i,j} for 1 <= i < j <= N
    # We store them in a dictionary for O(1) access: {(i, j): cost}
    # We use a list comprehension to flatten the remaining input and 
    # a dictionary comprehension to map pairs to costs.
    all_A = list(gen)
    
    # To map the flat list of A values to (i, j) pairs:
    # Pair (1,2), (1,3)...(1,N), (2,3)...(2,N), etc.
    pairs = [
        (i, j) 
        for i in range(1, N + 1) 
        for j in range(i + 1, N + 1)
    ]
    cost_map = {pair: cost for pair, cost in zip(pairs, all_A)}

    # Adjacency sets for quick lookup
    # G_adj contains edges of G
    G_adj = {edge for edge in G_edges}
    # H_adj contains edges of H
    H_adj = {edge for edge in H_edges}

    # We need to find a permutation P of (1...N) such that 
    # transforming H to be isomorphic to G via P is minimized.
    # The cost for a permutation P is the sum of A_{i,j} for all pairs (i,j)
    # where the existence of an edge (P_i, P_j) in H differs from 
    # the existence of an edge (i, j) in G.
    
    # Generate all permutations of vertices 1...N
    all_perms = permutations(range(1, N + 1))
    
    # For a fixed permutation P, the cost is:
    # Sum_{1 <= i < j <= N} A_{i,j} if (edge(i,j) in G) != (edge(P_i, P_j) in H)
    
    # We can pre-calculate the pairs (i, j) for the summation
    idx_pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    
    # Use a generator expression inside min() to find the minimum cost across all permutations
    # P is treated as a 0-indexed tuple, so vertex i is P[i-1]
    ans = min(
        sum(
            cost_map[(i, j)] 
            for i, j in idx_pairs 
            if ((i, j) in G_adj) != (tuple(sorted((P[i-1], P[j-1])))) in H_adj
        )
        for P in all_perms
    )
    
    print(ans)

if __name__ == "__main__":
    solve()