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
    # There are N*(N-1)//2 costs. We map them to a dictionary keyed by (i, j) where i < j.
    # The costs are provided in the order: A_{1,2}, A_{1,3}... A_{1,N}, A_{2,3}... A_{N-1,N}
    costs_list = list(gen)
    
    # To map the flat costs list to (i, j) pairs, we generate the pairs in the same order
    all_pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    cost_map = dict(zip(all_pairs, costs_list))

    # Represent graphs as sets of edges for O(1) lookup
    set_G = set(edges_G)
    set_H = set(edges_H)

    # We need to find a permutation P of (1...N) that minimizes the cost.
    # The cost for a permutation P is the sum of A_{P_i, P_j} for all pairs (i, j) 
    # where the edge status in G (i, j) differs from the edge status in H (P_i, P_j).
    
    # Pre-calculate all possible pairs (i, j) in G
    g_pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]

    # The cost function for a specific permutation P
    # P is a tuple where P[i-1] is the vertex in H corresponding to vertex i in G.
    def calculate_cost(P):
        # For every pair (i, j) in G, check if the edge exists in G and if the 
        # corresponding edge (P[i-1], P[j-1]) exists in H.
        # If they differ, add the cost A_{min(P_i, P_j), max(P_i, P_j)}.
        
        # We use a generator expression inside sum()
        return sum(
            cost_map[tuple(sorted((P[i-1], P[j-1])))]
            for i, j in g_pairs
            if ((i, j) in set_G) != (tuple(sorted((P[i-1], P[j-1])))) in set_H
        )

    # Try all N! permutations and find the minimum cost
    # N <= 8, so 8! = 40,320, which is well within time limits.
    ans = min(calculate_cost(P) for P in permutations(range(1, N + 1)))
    
    print(ans)

if __name__ == "__main__":
    solve()