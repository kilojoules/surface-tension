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
    
    # Graph G edges
    MG = int(next(it))
    G_edges = set()
    for _ in range(MG):
        u = int(next(it))
        v = int(next(it))
        # Store edges as sorted tuples to ensure consistency
        G_edges.add(tuple(sorted((u, v))))
        
    # Graph H edges
    MH = int(next(it))
    H_edges = set()
    for _ in range(MH):
        u = int(next(it))
        v = int(next(it))
        H_edges.add(tuple(sorted((u, v))))
        
    # Cost matrix A_{i,j}
    # The input gives A_{i,j} for 1 <= i < j <= N
    # We store them in a dictionary where keys are sorted tuples (i, j)
    costs = {}
    for i in range(1, N):
        for j in range(i + 1, N + 1):
            costs[(i, j)] = int(next(it))

    # We need to find a permutation P of (1, ..., N) such that 
    # the cost of transforming H to be isomorphic to G via P is minimized.
    # The cost for a permutation P is the sum of A_{P_i, P_j} for all pairs (i, j)
    # where the existence of an edge (i, j) in G differs from the existence 
    # of an edge (P_i, P_j) in H.
    
    # Pre-calculate all possible pairs (i, j) with i < j
    all_pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    
    # Function to calculate cost for a specific permutation P
    # P is a tuple where P[i-1] is the image of vertex i
    def calculate_cost(P):
        total_cost = 0
        # For every pair of vertices (i, j) in G
        for i, j in all_pairs:
            # Map them to vertices in H using the permutation P
            # P is 0-indexed, so we use P[i-1] and P[j-1]
            pi = P[i-1]
            pj = P[j-1]
            
            # Ensure the pair in H is sorted for lookup
            h_pair = tuple(sorted((pi, pj)))
            
            # Check if edge existence differs between G and H
            # (i, j) in G_edges XOR (pi, pj) in H_edges
            if ((i, j) in G_edges) != (h_pair in H_edges):
                total_cost += costs[h_pair]
        return total_cost

    # Try all permutations of {1, ..., N}
    # N <= 8, so N! <= 40320, which is feasible.
    min_total_cost = min(calculate_cost(p) for p in permutations(range(1, N + 1)))
    
    print(min_total_cost)

if __name__ == "__main__":
    solve()