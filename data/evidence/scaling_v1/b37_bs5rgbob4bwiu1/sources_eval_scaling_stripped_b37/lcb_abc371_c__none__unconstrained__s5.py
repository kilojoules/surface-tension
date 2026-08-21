import sys
from itertools import permutations

def solve():
    # Read all input at once and split into a list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use a generator/iterator to consume input tokens
    it = iter(input_data)
    
    N = int(next(it))
    
    # Read G edges
    MG = int(next(it))
    # Use a set of frozensets for efficient edge lookup in G
    # We use frozenset({u, v}) because the graph is undirected
    G_edges = {frozenset([int(next(it)), int(next(it))]) for _ in range(MG)}
    
    # Read H edges
    MH = int(next(it))
    H_edges = {frozenset([int(next(it)), int(next(it))]) for _ in range(MH)}
    
    # Read A_{i,j} costs
    # The input provides A_{i,j} for 1 <= i < j <= N
    # We store them in a dictionary keyed by frozenset({i, j})
    costs = {}
    for i in range(1, N):
        for j in range(i + 1, N + 1):
            costs[frozenset([i, j])] = int(next(it))
            
    # To make G and H isomorphic, we need a permutation P of {1...N}
    # such that edge (i, j) exists in G iff edge (P_i, P_j) exists in H.
    # The cost for a specific permutation P is the sum of A_{P_i, P_j} 
    # for all pairs (i, j) where the existence of edge (i, j) in G 
    # differs from the existence of edge (P_i, P_j) in H.
    
    # Pre-calculate all possible pairs (i, j) with 1 <= i < j <= N
    all_pairs = [frozenset([i, j]) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    
    # We iterate through all permutations of (1, ..., N)
    # For each permutation P, we calculate the total cost.
    # P maps vertex i in G to vertex P[i-1] in H.
    
    # We use a helper function to calculate cost for a given permutation
    # To optimize, we use a list comprehension inside sum()
    # We map the permutation to a tuple/list and use indices to access it.
    
    # Since N is small (up to 8), 8! = 40,320 is small enough for a brute force.
    
    # We define a function that takes the permutation tuple P
    # and calculates the cost. 
    # Note: P is a tuple where P[i-1] is the image of vertex i.
    def get_cost(P):
        # For every pair (i, j) in G, check if edge exists in G and if 
        # edge (P[i-1], P[j-1]) exists in H.
        # If they differ, add the cost A_{P[i-1], P[j-1]}.
        return sum(
            costs[frozenset([P[i-1], P[j-1]])]
            for i in range(1, N + 1)
            for j in range(i + 1, N + 1)
            if (frozenset([i, j]) in G_edges) != (frozenset([P[i-1], P[j-1]]) in H_edges)
        )

    # Generate all permutations of 1...N
    perms = permutations(range(1, N + 1))
    
    # Find the minimum cost across all permutations
    ans = min(get_cost(p) for p in perms)
    
    print(ans)

if __name__ == "__main__":
    solve()