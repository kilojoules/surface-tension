import itertools
import sys

def solve():
    # Read all input at once
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
    g_edges_raw = [int(next(it)) for _ in range(2 * M_G)]
    # Represent G as an adjacency matrix (set of frozen sets for fast lookup)
    # We use a set of tuples (i, j) where i < j
    G_adj = {tuple(sorted((g_edges_raw[i], g_edges_raw[i+1]))) 
             for i in range(0, 2 * M_G, 2)}
    
    # M_H: Number of edges in H
    M_H = int(next(it))
    # H edges
    h_edges_raw = [int(next(it)) for _ in range(2 * M_H)]
    H_adj = {tuple(sorted((h_edges_raw[i], h_edges_raw[i+1]))) 
             for i in range(0, 2 * M_H, 2)}
    
    # A_{i,j}: Costs
    # The costs are provided in a specific triangular format:
    # A_{1,2}, A_{1,3}... A_{1,N}
    # A_{2,3}... A_{2,N}
    # ...
    # A_{N-1,N}
    # We map these to a dictionary {(i, j): cost} where i < j
    costs_raw = [int(next(it)) for _ in range(N * (N - 1) // 2)]
    
    # To map the flat costs list to (i, j) pairs:
    # Pair (i, j) with 1 <= i < j <= N
    all_pairs = [tuple(sorted((i, j))) 
                 for i in range(1, N + 1) 
                 for j in range(i + 1, N + 1)]
    
    cost_map = dict(zip(all_pairs, costs_raw))
    
    # We need to find a permutation P of (1, ..., N) that minimizes the cost.
    # The cost for a permutation P is the sum of A_{P_i, P_j} for all pairs (i, j)
    # such that the edge status of (i, j) in G is different from (P_i, P_j) in H.
    # Wait, the problem says: "make G and H isomorphic".
    # This means we look for a permutation P such that we modify H to match G.
    # Specifically, for all i < j, edge (i, j) exists in G iff edge (P_i, P_j) exists in H.
    # If they differ, we pay A_{P_i, P_j}.
    
    # Pre-calculate all pairs (i, j) with i < j
    pairs = list(itertools.combinations(range(1, N + 1), 2))
    
    # We want to minimize sum_{i<j} cost(P_i, P_j) if (edge(i,j) in G != edge(P_i, P_j) in H)
    # Let's use a helper to calculate cost for a given permutation
    def get_cost(p):
        # p is a permutation of (1, ..., N)
        # We map vertex i in G to vertex p[i-1] in H
        # For every pair (i, j) in G, check if edge exists. 
        # Compare with edge (p[i-1], p[j-1]) in H.
        # If different, add cost A_{p[i-1], p[j-1]}
        
        # Using a generator expression inside sum()
        return sum(
            cost_map[tuple(sorted((p[i-1], p[j-1])))]
            for i, j in pairs
            if ((i, j) in G_adj) != (tuple(sorted((p[i-1], p[j-1]))) in H_adj)
        )

    # Try all N! permutations
    # N <= 8, so 8! = 40,320, which is feasible.
    all_perms = itertools.permutations(range(1, N + 1))
    print(min(get_cost(p) for p in all_perms))

if __name__ == "__main__":
    solve()