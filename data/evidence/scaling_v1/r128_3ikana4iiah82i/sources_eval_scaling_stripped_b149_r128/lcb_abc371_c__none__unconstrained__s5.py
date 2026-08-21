import sys
from itertools import permutations

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
    # We map them to a dictionary where keys are sorted tuples (i, j)
    # The input order is A_{1,2}, A_{1,3}... A_{1,N}, A_{2,3}... A_{2,N}, etc.
    costs_list = list(gen)
    
    # Construct the cost mapping based on the specified input order
    # We generate the keys in the exact order they appear in the input
    cost_keys = [
        (i, j) 
        for i in range(1, N + 1) 
        for j in range(i + 1, N + 1)
    ]
    cost_map = dict(zip(cost_keys, costs_list))

    # Helper to check if an edge exists in a set of edges
    # We use sets for O(1) lookup
    set_G = set(edges_G)
    set_H = set(edges_H)

    # We need to find a permutation P of (1...N) that minimizes the cost.
    # The cost for a permutation P is the sum of A_{P_i, P_j} for all pairs (i, j)
    # where the edge status in G (i, j) differs from the edge status in H (P_i, P_j).
    
    # Generate all possible permutations of vertices 1...N
    all_perms = permutations(range(1, N + 1))
    
    # For each permutation, calculate the total cost
    # We iterate through all possible edges (i, j) in G
    # and check if the corresponding edge (P_i, P_j) in H matches.
    
    # To avoid loops inside the comprehension, we pre-generate all possible edges in G
    all_possible_edges = [
        (i, j) 
        for i in range(1, N + 1) 
        for j in range(i + 1, N + 1)
    ]

    # The cost function for a single permutation P:
    # For every pair (i, j), if (edge in G) != (edge in H after permutation), add cost.
    # Note: P is a tuple where P[i-1] is the image of vertex i.
    
    def calculate_cost(P):
        # P is 0-indexed, so vertex i is at P[i-1]
        # For every pair of vertices i, j in G:
        # The corresponding vertices in H are P[i-1] and P[j-1].
        # We need to sort the pair (P[i-1], P[j-1]) to look it up in the cost map.
        
        # We use a generator expression inside sum()
        return sum(
            cost_map[tuple(sorted((P[i-1], P[j-1])))]
            for i, j in all_possible_edges
            if ( (i, j) in set_G ) != ( tuple(sorted((P[i-1], P[j-1]))) in set_H )
        )

    # Find the minimum cost across all permutations
    ans = min(calculate_cost(P) for P in all_perms)
    print(ans)

if __name__ == "__main__":
    solve()