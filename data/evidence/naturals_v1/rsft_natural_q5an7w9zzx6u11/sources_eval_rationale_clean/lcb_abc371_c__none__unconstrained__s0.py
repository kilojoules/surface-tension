import sys
from itertools import permutations

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to consume the input stream
    it = iter(input_data)
    
    # N is the number of vertices
    N = next(it)
    
    # Read edges of G
    MG = next(it)
    # Create an adjacency matrix for G
    # G_adj[i][j] == 1 if edge exists, else 0
    # We use a dictionary with tuple keys for easy lookup
    G_edges = set([tuple(sorted((next(it), next(it)))) for _ in range(MG)])
    
    # Read edges of H
    MH = next(it)
    H_edges = set([tuple(sorted((next(it), next(it)))) for _ in range(MH)])
    
    # Read costs A_{i,j}
    # The costs are provided in a specific order: (1,2), (1,3)...(1,N), (2,3)...(N-1,N)
    # We store them in a dictionary keyed by sorted vertex pairs
    def get_cost_pairs(n):
        return [(i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1)]
    
    costs_list = [next(it) for _ in range(N * (N - 1) // 2)]
    cost_map = dict(zip(get_cost_pairs(N), costs_list))
    
    # Two graphs are isomorphic if there exists a permutation P such that
    # edge (i, j) in G <=> edge (P[i], P[j]) in H.
    # We want to minimize the cost to make H isomorphic to G.
    # For a fixed permutation P, the cost is the sum of A_{P[i], P[j]} 
    # for all pairs (i, j) where the edge status in G differs from H.
    
    # Pre-calculate all possible pairs (i, j) with 1 <= i < j <= N
    all_pairs = get_cost_pairs(N)
    
    # We iterate through all permutations of (1, ..., N)
    # P[i-1] is the vertex in H that vertex i in G is mapped to.
    def calculate_total_cost(P):
        # For every pair (i, j) in G, check if the corresponding pair (P[i-1], P[j-1]) in H
        # has the same edge status. If not, add the cost A_{P[i-1], P[j-1]}.
        # Note: P is a tuple, so P[i-1] is the vertex label.
        return sum(
            cost_map[tuple(sorted((P[i-1], P[j-1])))]
            for (i, j) in all_pairs
            if ((i, j) in G_edges) != (tuple(sorted((P[i-1], P[j-1]))) in H_edges)
        )

    # Generate all permutations and find the minimum cost
    # Using a generator expression inside min() to avoid creating a large list in memory
    result = min(calculate_total_cost(P) for P in permutations(range(1, N + 1)))
    
    print(result)

if __name__ == "__main__":
    solve()