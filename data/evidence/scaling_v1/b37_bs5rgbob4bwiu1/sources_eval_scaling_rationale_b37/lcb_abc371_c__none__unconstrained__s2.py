import sys
from itertools import permutations

def solve():
    # Read all input at once and split into a list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract values one by one
    gen = input_data
    
    try:
        N = next(gen)
    except StopIteration:
        return

    # Graph G edges
    MG = next(gen)
    # Use a set of frozensets for fast edge lookup in G
    G_edges = {frozenset([next(gen), next(gen)]) for _ in range(MG)}
    
    # Graph H edges
    MH = next(gen)
    H_edges = {frozenset([next(gen), next(gen)]) for _ in range(MH)}
    
    # Cost matrix A
    # A[i][j] will store the cost to toggle edge (i+1, j+1)
    # We read the remaining values into a flat list first
    A_flat = list(gen)
    
    # To access A_{i,j} without loops, we pre-calculate the mapping
    # Since N is small (<= 8), we can create a dictionary for costs
    # The input gives A_{1,2}, A_{1,3}... A_{1,N}, A_{2,3}...
    # We map (i, j) where i < j to the index in A_flat
    
    # Pre-calculate indices for the flat A list
    # Index for (i, j) is sum_{k=1}^{i-1} (N-k) + (j-i-1)
    def get_cost(i, j):
        if i > j: i, j = j, i
        # i and j are 1-indexed
        # The number of elements before row i is (N-1) + (N-2) + ... + (N-(i-1))
        # Which is (i-1)*N - (i-1)*i // 2
        idx = (i - 1) * N - (i * (i - 1)) // 2 + (j - i - 1)
        return A_flat[idx]

    # We need to find a permutation P of {1, ..., N} that minimizes cost
    # Cost for a permutation P:
    # Sum over all 1 <= i < j <= N of:
    # A_{P_i, P_j} if (edge (i,j) in G != edge (P_i, P_j) in H)
    
    # Pre-calculate all pairs (i, j) with 1 <= i < j <= N
    pairs = [ (i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1) ]
    
    # Function to calculate total cost for a specific permutation P
    def calculate_total_cost(P):
        # P is a tuple (P1, P2, ..., PN)
        # We need to check if edge (i, j) in G has the same status as (P_{i-1}, P_{j-1}) in H
        return sum(
            get_cost(P[i-1], P[j-1]) 
            for i, j in pairs 
            if (frozenset([i, j]) in G_edges) != (frozenset([P[i-1], P[j-1]]) in H_edges)
        )

    # Iterate over all permutations of (1, ..., N) and find the minimum cost
    # Using map and min instead of a for loop
    ans = min(map(calculate_total_cost, permutations(range(1, N + 1))))
    
    print(ans)

if __name__ == "__main__":
    solve()