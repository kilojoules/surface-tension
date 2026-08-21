import itertools
import sys

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use a generator/iterator to consume input
    it = iter(input_data)
    
    # N: Number of vertices
    N = int(next(it))
    
    # M_G: Number of edges in G
    M_G = int(next(it))
    # G edges
    g_edges_list = [
        (int(next(it)), int(next(it))) 
        for _ in range(M_G)
    ]
    
    # M_H: Number of edges in H
    M_H = int(next(it))
    # H edges
    h_edges_list = [
        (int(next(it)), int(next(it))) 
        for _ in range(M_H)
    ]
    
    # A_{i,j}: Costs
    # The costs are provided in a specific triangular format:
    # A_{1,2}, A_{1,3}... A_{1,N}
    # A_{2,3}... A_{2,N}
    # ...
    # A_{N-1,N}
    # We map these into a dictionary for O(1) access: cost[(i, j)] where i < j
    costs_flat = [int(next(it)) for _ in range(N * (N - 1) // 2)]
    
    # To map the flat list to (i, j) pairs:
    # Row 1 has N-1 elements, Row 2 has N-2, etc.
    cost_map = {}
    current_idx = 0
    for i in range(1, N):
        for j in range(i + 1, N + 1):
            cost_map[(i, j)] = costs_flat[current_idx]
            current_idx += 1

    # Represent G and H as adjacency matrices (sets of frozen sets for fast lookup)
    # Using sets of tuples (min(u,v), max(u,v))
    g_set = {tuple(sorted(edge)) for edge in g_edges_list}
    h_set = {tuple(sorted(edge)) for edge in h_edges_list}

    # We need to find a permutation P of (1...N) that minimizes the cost.
    # The cost for a permutation P is the sum of A_{P_i, P_j} for all pairs (i, j)
    # where the existence of edge (i, j) in G differs from the existence of edge (P_i, P_j) in H.
    
    # Pre-calculate all pairs (i, j) with i < j
    all_pairs = list(itertools.combinations(range(1, N + 1), 2))
    
    # We want to minimize:
    # Sum_{i < j} [ (edge(i,j) in G) != (edge(P_i, P_j) in H) ] * A_{P_i, P_j}
    
    # Since N is small (up to 8), we can iterate through all N! permutations.
    # 8! = 40,320, which is well within limits for Python if we use a comprehension.
    
    # We use a helper to get the cost for a specific permutation P.
    # P is a tuple where P[i-1] is the image of vertex i.
    
    def calculate_cost(P):
        # For every pair (i, j) in G, check if the corresponding pair (P_i, P_j) in H
        # has the same edge status. If not, add the cost A_{P_i, P_j}.
        # Note: P is 0-indexed, so vertex i is at P[i-1].
        
        # We iterate over all pairs (i, j) and check if G(i,j) == H(P_i, P_j)
        # The cost is associated with the vertices in H, so we use A_{min(P_i, P_j), max(P_i, P_j)}
        
        # Using a generator expression inside sum()
        return sum(
            cost_map[tuple(sorted((P[i-1], P[j-1])))]
            for i, j in all_pairs
            if ((i, j) in g_set) != (tuple(sorted((P[i-1], P[j-1]))) in h_set)
        )

    # Try all permutations of (1...N)
    # We use a list comprehension to find the minimum cost across all permutations.
    ans = min(
        calculate_cost(P) 
        for P in itertools.permutations(range(1, N + 1))
    )
    
    print(ans)

if __name__ == "__main__":
    solve()