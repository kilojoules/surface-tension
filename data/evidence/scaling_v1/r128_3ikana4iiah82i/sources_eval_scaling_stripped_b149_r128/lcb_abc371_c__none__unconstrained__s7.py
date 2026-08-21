import sys
from itertools import permutations
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    ptr = 0
    N = int(input_data[ptr])
    ptr += 1
    
    # M_G and edges of G
    M_G = int(input_data[ptr])
    ptr += 1
    edges_G = []
    for _ in range(M_G):
        u = int(input_data[ptr])
        v = int(input_data[ptr+1])
        edges_G.append(tuple(sorted((u, v))))
        ptr += 2
        
    # M_H and edges of H
    M_H = int(input_data[ptr])
    ptr += 1
    edges_H = []
    for _ in range(M_H):
        u = int(input_data[ptr])
        v = int(input_data[ptr+1])
        edges_H.append(tuple(sorted((u, v))))
        ptr += 2
        
    # A_{i,j} costs
    # The costs are provided for 1 <= i < j <= N
    # We store them in a dictionary keyed by (i, j)
    costs_list = input_data[ptr:]
    
    # To map the flat list of costs to (i, j) pairs, we generate the pairs in order
    # i from 1 to N-1, j from i+1 to N
    cost_pairs = [(i, j) for i in range(1, N) for j in range(i + 1, N)]
    cost_map = dict(zip(cost_pairs, map(int, costs_list)))

    # Helper to check if an edge exists in a set of edges
    # We use sets for O(1) lookup
    set_G = set(edges_G)
    set_H = set(edges_H)

    # The core logic: for a given permutation P, calculate the cost to make H isomorphic to G.
    # An edge (i, j) in G must correspond to an edge (P[i-1], P[j-1]) in H.
    # If the edge exists in G but not in H, we pay to add it to H.
    # If the edge exists in H but not in G, we pay to remove it from H.
    # Note: the cost is based on the vertices in H, so we use sorted(P[i-1], P[j-1]).
    
    def calculate_cost(P):
        # We iterate over all possible edges (i, j) with 1 <= i < j <= N
        # and check if the isomorphism condition is violated.
        return sum(
            cost_map[tuple(sorted((P[i-1], P[j-1])))]
            for i in range(1, N)
            for j in range(i + 1, N)
            if ((i, j) in set_G) != (tuple(sorted((P[i-1], P[j-1]))) in set_H)
        )

    # We need to find the minimum cost across all permutations of (1, ..., N)
    # Since N <= 8, N! is at most 40,320, which fits within time limits.
    all_permutations = permutations(range(1, N + 1))
    
    # Use map to apply calculate_cost to all permutations and min to find the result
    # We use a generator expression inside min()
    result = min(calculate_cost(P) for P in all_permutations)
    
    print(result)

if __name__ == "__main__":
    solve()