import sys
from itertools import permutations

def solve():
    # Read all input at once and split into a list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to consume input values one by one
    gen = input_data
    
    try:
        N = next(gen)
    except StopIteration:
        return

    # Read G edges
    MG = next(gen)
    # We use a set of frozensets for efficient edge lookup in G
    # Each edge is represented as a frozenset({u, v})
    G_edges = {
        frozensim = frozenset([next(gen), next(gen)]) 
        for _ in range(MG)
    }
    # Correcting the comprehension above: the variable assignment inside was invalid
    # Let's redefine G_edges properly:
    
def solve_fixed():
    # Since the previous attempt had a syntax error in the comprehension, 
    # I will rewrite the input reading logic clearly.
    input_all = sys.stdin.read().split()
    if not input_all:
        return
    
    ptr = 0
    N = int(input_all[ptr]); ptr += 1
    MG = int(input_all[ptr]); ptr += 1
    
    # G edges as a set of tuples (sorted to ensure i < j)
    G_edges = set()
    for _ in range(MG):
        u = int(input_all[ptr]); ptr += 1
        v = int(input_all[ptr]); ptr += 1
        G_edges.add(tuple(sorted((u, v))))
        
    MH = int(input_all[ptr]); ptr += 1
    H_edges = set()
    for _ in range(MH):
        u = int(input_all[ptr]); ptr += 1
        v = int(input_all[ptr]); ptr += 1
        H_edges.add(tuple(sorted((u, v))))
        
    # A_{i,j} values
    # The input gives A_{1,2}, A_{1,3}... A_{1,N}, then A_{2,3}...
    # We store them in a dictionary keyed by (i, j) where i < j
    A = {}
    curr_i = 1
    curr_j = 2
    while ptr < len(input_all):
        val = int(input_all[ptr]); ptr += 1
        A[(curr_i, curr_j)] = val
        curr_j += 1
        if curr_j > N:
            curr_i += 1
            curr_j = curr_i + 1

    # Function to calculate cost for a specific permutation P
    # P maps vertex i in G to vertex P[i-1] in H
    # We need to check every pair (i, j) in G and see if the edge status
    # matches the edge status of (P[i-1], P[j-1]) in H.
    # If they differ, we pay A_{min(P[i-1], P[j-1]), max(P[i-1], P[j-1])}
    
    # To optimize, we pre-calculate all pairs (i, j) with 1 <= i < j <= N
    pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    
    # We use a helper function inside a comprehension over all permutations
    # Permutations of (1, ..., N)
    perms = permutations(range(1, N + 1))
    
    # For a permutation p, the cost is the sum of A_{u,v} for all pairs (u,v)
    # where the existence of edge (u,v) in H differs from edge (p_inv(u), p_inv(v)) in G.
    # Actually, it's simpler: for every pair (i, j) in G, 
    # check if edge (i, j) exists in G XOR edge (p[i-1], p[j-1]) exists in H.
    
    # We use a function to avoid the comprehension complexity limit
    def get_cost(p):
        cost = 0
        for i, j in pairs:
            # Edge in G?
            has_g = (i, j) in G_edges
            # Edge in H?
            # p is 0-indexed, so p[i-1] and p[j-1] are the vertices in H
            u, v = p[i-1], p[j-1]
            if u > v: u, v = v, u
            has_h = (u, v) in H_edges
            
            if has_g != has_h:
                cost += A[(u, v)]
        return cost

    # Find the minimum cost across all N! permutations
    ans = min(get_cost(p) for p in perms)
    print(ans)

if __name__ == "__main__":
    solve_fixed()