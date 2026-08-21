import sys
from itertools import permutations

def solve():
    # Read all input at once and split into a list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Using a generator to consume input sequentially
    gen = input_data
    
    def next_val():
        return next(gen)

    try:
        N = next_val()
    except StopIteration:
        return

    # Graph G edges
    MG = next_val()
    G_edges = set()
    for _ in range(MG):
        u, v = next_val(), next_val()
        G_edges.add(tuple(sorted((u, v))))

    # Graph H edges
    MH = next_val()
    H_edges = set()
    for _ in range(MH):
        u, v = next_val(), next_val()
        H_edges.add(tuple(sorted((u, v))))

    # Cost matrix A_{i,j}
    # The input provides A_{i,j} for 1 <= i < j <= N
    # We store them in a dictionary with sorted tuple keys for easy lookup
    costs = {}
    for i in range(1, N):
        for j in range(i + 1, N + 1):
            costs[(i, j)] = next_val()

    # We need to find a permutation P of (1...N) that minimizes the cost.
    # The cost for a permutation P is the sum of A_{P_i, P_j} for all pairs (i, j)
    # where the edge status in G(i, j) differs from the edge status in H(P_i, P_j).
    # Note: The problem defines isomorphism as: edge (i, j) in G <=> edge (P_i, P_j) in H.
    # So we iterate over all pairs 1 <= i < j <= N.
    
    # Pre-calculate all pairs (i, j) for G
    all_pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    
    # Function to calculate cost for a specific permutation P
    # P is a tuple where P[i-1] is the image of vertex i
    def calculate_cost(P):
        # P is 0-indexed, so vertex i is at P[i-1]
        # For every pair (i, j) in G, check if edge exists in G and if edge exists in H between P[i-1] and P[j-1]
        # If they differ, add the cost A_{min(P[i-1], P[j-1]), max(P[i-1], P[j-1])}
        
        # We use a generator expression inside sum() for efficiency
        return sum(
            costs[tuple(sorted((P[i-1], P[j-1])))]
            for i, j in all_pairs
            if ((i, j) in G_edges) != (tuple(sorted((P[i-1], P[j-1]))) in H_edges)
        )

    # Try all permutations of (1...N)
    # N <= 8, so N! <= 40320, which is feasible in Python.
    min_total_cost = min(
        calculate_cost(p) 
        for p in permutations(range(1, N + 1))
    )

    print(min_total_cost)

if __name__ == "__main__":
    solve()