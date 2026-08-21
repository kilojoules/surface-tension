import itertools
import sys

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    it = iter(input_data)
    N = int(next(it))
    
    # Graph G edges
    MG = int(next(it))
    G_edges = [tuple(sorted((int(next(it)), int(next(it))))) for _ in range(MG)]
    
    # Graph H edges
    MH = int(next(it))
    H_edges = [tuple(sorted((int(next(it)), int(next(it))))) for _ in range(MH)]
    
    # Cost matrix A
    # A is provided as A_{1,2}, A_{1,3}... A_{2,3}...
    # We store it in a dictionary for easy lookup: (i, j) -> cost
    all_costs = [int(next(it)) for _ in range(N * (N - 1) // 2)]
    
    # Map the flat cost list to pairs (i, j)
    # The order of pairs is (1,2), (1,3)...(1,N), (2,3)...(2,N), ..., (N-1, N)
    pairs = [ (i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1) ]
    cost_map = dict(zip(pairs, all_costs))

    # Adjacency matrices for G and H
    # Using sets of tuples for O(1) lookup
    G_set = set(G_edges)
    H_set = set(H_edges)

    # We need to find a permutation P of (1...N) such that 
    # the cost to make H isomorphic to G via P is minimized.
    # The cost for a permutation P is the sum of A_{P_i, P_j} 
    # for all pairs (i, j) where (edge in G) != (edge in H after permutation).
    # Specifically, if edge (i, j) exists in G, we need edge (P_i, P_j) in H.
    # If it doesn't, we need (P_i, P_j) to not exist in H.
    
    # Pre-calculate all pairs (i, j) for G
    all_pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    
    # The cost to adjust H for a specific permutation P:
    # For every pair (i, j) in G:
    #   Let u = P[i-1], v = P[j-1]. 
    #   Let edge_G = (i, j) in G_set
    #   Let edge_H = (min(u, v), max(u, v)) in H_set
    #   If edge_G != edge_H, add cost_map[(min(u, v), max(u, v))]
    
    # To avoid loops, we use a generator expression inside min()
    # and a list comprehension to handle the permutation mapping.
    
    ans = min(
        sum(
            cost_map[(min(p[i-1], p[j-1]), max(p[i-1], p[j-1]))]
            for i, j in all_pairs
            if ((i, j) in G_set) != ((min(p[i-1], p[j-1]), max(p[i-1], p[j-1])) in H_set)
        )
        for p in itertools.permutations(range(1, N + 1))
    )
    
    print(ans)

if __name__ == "__main__":
    solve()