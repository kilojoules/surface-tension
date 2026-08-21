import sys
from itertools import permutations

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract values one by one
    gen = input_data
    
    N = next(gen)
    
    # Helper to read M edges and return a set of frozensets (undirected edges)
    def read_edges(m_val):
        return {frozenset([next(gen), next(gen)]) for _ in range(m_val)}

    M_G = next(gen)
    edges_G = read_edges(M_G)
    
    M_H = next(gen)
    edges_H = read_edges(M_H)
    
    # Read the cost matrix A_{i,j}
    # A is provided as a triangular matrix. We map it to a dictionary for easy lookup.
    # The input format is A_{1,2}, A_{1,3}... A_{1,N}, A_{2,3}... A_{N-1,N}
    def parse_costs(n):
        # We need to map (i, j) where i < j to the cost.
        # There are N*(N-1)//2 such pairs.
        all_costs = [next(gen) for _ in range(n * (n - 1) // 2)]
        
        # To avoid loops, we can use a list comprehension to create the mapping.
        # The index in all_costs corresponds to the pair (i, j).
        # Pair (i, j) index = (i-1)*N - (i*(i+1)//2) + (j-i-1) is too complex.
        # Simpler: use a comprehension to generate all (i, j) pairs and zip them.
        pairs = [(i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1)]
        return dict(zip(pairs, all_costs))

    costs_dict = parse_costs(N)

    # We want to find a permutation P of {1...N} that minimizes the cost.
    # Cost for a permutation P:
    # For every pair (i, j) with 1 <= i < j <= N:
    # If (edge exists in G between i,j) != (edge exists in H between P_i, P_j):
    #     add cost A_{P_i, P_j}
    
    # Pre-calculate all pairs (i, j) for G
    g_pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    
    def calculate_cost(p):
        # p is a permutation of (1, ..., N)
        # We map vertex i in G to vertex p[i-1] in H.
        # For every pair (i, j) in G, we check if the edge status matches in H.
        # The cost is associated with the vertices in H: A_{min(p[i-1], p[j-1]), max(p[i-1], p[j-1])}
        
        return sum(
            costs_dict[(min(p[i-1], p[j-1]), max(p[i-1], p[j-1]))]
            for i, j in g_pairs
            if (frozenset([i, j]) in edges_G) != (frozenset([p[i-1], p[j-1]]) in edges_H)
        )

    # Try all permutations of (1, ..., N) and find the minimum cost
    ans = min(map(calculate_cost, permutations(range(1, N + 1))))
    print(ans)

if __name__ == "__main__":
    solve()