import itertools
import sys

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to consume input sequentially
    gen = input_data
    
    try:
        N = next(gen)
    except StopIteration:
        return

    # Read edges of G
    M_G = next(gen)
    edges_G = [tuple(sorted((next(gen), next(gen)))) for _ in range(M_G)]
    
    # Read edges of H
    M_H = next(gen)
    edges_H = [tuple(sorted((next(gen), next(gen)))) for _ in range(M_H)]
    
    # Read costs A_{i,j}
    # There are N*(N-1)//2 costs provided in the input
    all_costs = list(gen)
    
    # Map each pair (i, j) with i < j to its cost
    # The costs are provided in the order: (1,2), (1,3)...(1,N), (2,3)...(2,N), etc.
    # We can pre-calculate the indices or just use a dictionary
    cost_map = {}
    idx = 0
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            cost_map[(i, j)] = all_costs[idx]
            idx += 1

    # Convert edge lists to sets for O(1) lookup
    set_G = set(edges_G)
    set_H = set(edges_H)

    # Generate all possible permutations of vertices (1...N)
    # A permutation P maps vertex i in G to vertex P[i-1] in H
    perms = itertools.permutations(range(1, N + 1))

    # For each permutation, calculate the cost to make H isomorphic to G
    # The cost is the sum of A_{P_i, P_j} for all pairs (i, j) where:
    # (edge exists in G AND NOT in H) OR (edge NOT in G AND exists in H)
    # This is equivalent to: edge_in_G XOR edge_in_H
    
    # We pre-calculate all possible pairs (i, j) to avoid loops inside the comprehension
    all_pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]

    # The objective is to find the minimum cost across all permutations
    # For a permutation P, the cost is sum(cost_map[sorted(P[i-1], P[j-1])] 
    # for (i, j) in all_pairs if (i, j) in set_G != (P[i-1], P[j-1]) in set_H)
    
    # To optimize, we use a helper function for the cost of a single permutation
    def get_cost(p):
        # p is a tuple (P_1, P_2, ..., P_N)
        # We need to check every pair (i, j) in G and see if the corresponding 
        # pair (p[i-1], p[j-1]) in H matches the edge status.
        
        # Create a list of costs for pairs that differ in edge status
        # We use sorted() to ensure the key for cost_map is always (smaller, larger)
        return sum(
            cost_map[tuple(sorted((p[i-1], p[j-1])))]
            for i, j in all_pairs
            if ((i, j) in set_G) != (tuple(sorted((p[i-1], p[j-1]))) in set_H)
        )

    # Use map to apply get_cost to all permutations and find the minimum
    print(min(map(get_cost, perms)))

if __name__ == "__main__":
    solve()