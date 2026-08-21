import sys
from itertools import permutations

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = map(int, sys.stdin.read().split())
    # Use a generator to consume the input stream
    gen = iter(input_data)
    
    N = next(gen)
    
    # Read Graph G edges
    Mg = next(gen)
    g_edges = [tuple(sorted((next(gen), next(gen)))) for _ in range(Mg)]
    # Represent G as an adjacency matrix for O(1) lookup
    G = [[False] * (N + 1) for _ in range(N + 1)]
    # Using a list comprehension to populate the matrix without a for-loop
    # Since we can't use for-loops, we use a trick with a list and map/side-effects
    # However, the cleanest way to build the matrix is a nested comprehension
    G = [[(i, j) in g_edges or (j, i) in g_edges for j in range(1, N + 1)] for i in range(1, N + 1)]
    # Adjust G to be 0-indexed for easier permutation mapping
    G = [[(i+1, j+1) in g_edges or (j+1, i+1) in g_edges for j in range(N)] for i in range(N)]

    # Read Graph H edges
    Mh = next(gen)
    h_edges = [tuple(sorted((next(gen), next(gen)))) for _ in range(Mh)]
    H = [[(i+1, j+1) in h_edges or (j+1, i+1) in h_edges for j in range(N)] for i in range(N)]

    # Read Cost Matrix A
    # A is given as A_{1,2}, A_{1,3}... A_{N-1,N}
    # We need to map these to a structure we can index easily
    all_costs = list(gen)
    
    # To avoid loops, we pre-calculate the indices for the cost matrix
    # cost_map[(i, j)] = cost to flip edge between vertex i and j (0-indexed)
    # There are N*(N-1)//2 costs provided.
    # We can use a dictionary comprehension to map (i, j) pairs to their costs.
    cost_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    cost_dict = dict(zip(cost_pairs, all_costs))

    # The core logic:
    # For a permutation P, vertex i in G maps to vertex P[i] in H.
    # An edge (i, j) in G must exist in H between P[i] and P[j].
    # If G[i][j] != H[P[i]][P[j]], we pay cost A_{min(P[i], P[j]), max(P[i], P[j])}.
    
    # We use a generator expression inside min() to iterate over all permutations.
    # We use a sum() inside to calculate the total cost for a specific permutation.
    
    ans = min(
        sum(
            cost_dict[(min(p[i], p[j]), max(p[i], p[j]))]
            for i in range(N)
            for j in range(i + 1, N)
            if G[i][j] != H[p[i]][p[j]]
        )
        for p in permutations(range(N))
    )
    
    print(ans)

if __name__ == "__main__":
    solve()