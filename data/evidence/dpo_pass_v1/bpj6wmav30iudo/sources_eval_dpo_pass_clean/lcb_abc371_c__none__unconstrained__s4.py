import sys
from itertools import permutations

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use an iterator to consume the input data sequentially
    it = iter(input_data)
    
    # N vertices
    N = int(next(it))
    
    # Graph G edges
    MG = int(next(it))
    G_edges = [tuple(sorted((int(next(it)), int(next(it))))) for _ in range(MG)]
    
    # Graph H edges
    MH = int(next(it))
    H_edges = [tuple(sorted((int(next(it)), int(next(it))))) for _ in range(MH)]
    
    # Cost matrix A
    # A[i][j] where i < j
    # We store it in a dictionary for easy access: (i, j) -> cost
    # The input gives A_{1,2}, A_{1,3}... A_{1,N}, then A_{2,3}...
    # We can flatten the remaining input and map it to pairs (i, j)
    costs_flat = [int(x) for x in it]
    
    # Generate all pairs (i, j) with 1 <= i < j <= N in the order they appear in input
    all_pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    cost_map = dict(zip(all_pairs, costs_flat))
    
    # Adjacency matrices for G and H (using sets for O(1) lookup)
    G_set = set(G_edges)
    H_set = set(H_edges)
    
    # A permutation P maps vertex i in G to vertex P[i-1] in H.
    # G has an edge (i, j) iff H has an edge (P[i-1], P[j-1]).
    # If they differ, we pay the cost A_{min(P[i-1], P[j-1]), max(P[i-1], P[j-1])}.
    
    # We iterate through all permutations of (1, ..., N)
    # For each permutation, we calculate the total cost.
    # The result is the minimum of these costs.
    
    # To avoid loops, we use a generator expression inside min()
    # and a list comprehension to calculate the cost for a specific permutation.
    
    ans = min(
        sum(
            cost_map[tuple(sorted((p[i-1], p[j-1]))))]
            for i in range(1, N + 1)
            for j in range(i + 1, N + 1)
            if ((i, j) in G_set) != (tuple(sorted((p[i-1], p[j-1]))) in H_set)
        )
        for p in permutations(range(1, N + 1))
    )
    
    print(ans)

if __name__ == "__main__":
    solve()