import sys
from itertools import permutations

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use a pointer-like approach via an iterator to consume input
    it = iter(input_data)
    
    N = int(next(it))
    
    # Graph G edges
    MG = int(next(it))
    G_edges = [tuple(sorted((int(next(it)), int(next(it))))) for _ in range(MG)]
    
    # Graph H edges
    MH = int(next(it))
    H_edges = [tuple(sorted((int(next(it)), int(next(it))))) for _ in range(MH)]
    
    # Cost matrix A
    # A is given as A_{1,2}, A_{1,3}... A_{N-1,N}
    # We store it in a dictionary for O(1) access: {(i, j): cost} where i < j
    # We use a list comprehension to flatten the remaining input and map it
    costs_flat = [int(x) for x in it]
    
    # To map the flat list of costs to (i, j) pairs:
    # The pairs are (1,2), (1,3)...(1,N), (2,3)...(2,N), ..., (N-1, N)
    pairs = [ (i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1) ]
    cost_map = dict(zip(pairs, costs_flat))

    # Represent G and H as adjacency sets for fast lookup
    # Using sets of frozensets or sorted tuples
    g_set = set(G_edges)
    h_set = set(H_edges)

    # We want to find a permutation P of (1...N) such that 
    # we minimize the cost of changing H to match G under mapping P.
    # Specifically, for every pair (i, j) with i < j:
    # If (i, j) is an edge in G, then (P[i], P[j]) must be an edge in H.
    # If it isn't, we pay A_{P[i], P[j]}.
    # If (i, j) is NOT an edge in G, then (P[i], P[j]) must NOT be an edge in H.
    # If it is, we pay A_{P[i], P[j]}.
    
    # Pre-calculate all possible pairs (i, j) for the cost summation
    all_pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    
    # We iterate over all permutations of (1...N)
    # P is a tuple where P[i-1] is the vertex in H corresponding to vertex i in G.
    def calculate_cost(P):
        # For each pair (i, j) in G, check if the corresponding pair in H matches
        # The cost is incurred if the edge status differs.
        return sum(
            cost_map[tuple(sorted((P[i-1], P[j-1]))))]
            for i, j in all_pairs
            if ((i, j) in g_set) != (tuple(sorted((P[i-1], P[j-1]))) in h_set)
        )

    # Use min() with a generator expression to avoid explicit loops
    # permutations(range(1, N + 1)) generates all possible mappings
    ans = min(map(calculate_cost, permutations(range(1, N + 1))))
    
    print(ans)

if __name__ == "__main__":
    solve()