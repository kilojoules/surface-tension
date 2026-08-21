import sys
from itertools import permutations

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use a generator/iterator to consume input without indices
    it = iter(input_data)
    
    N = int(next(it))
    
    # Graph G edges
    MG = int(next(it))
    G_edges = [tuple(sorted((int(next(it)), int(next(it))))) for _ in range(MG)]
    
    # Graph H edges
    MH = int(next(it))
    H_edges = [tuple(sorted((int(next(it)), int(next(it))))) for _ in range(MH)]
    
    # Cost matrix A
    # A is provided as a flattened list of A_{i,j} for 1 <= i < j <= N
    # We store it in a dictionary for O(1) access: {(i, j): cost}
    A_flat = [int(next(it)) for _ in range(N * (N - 1) // 2)]
    
    # Create pairs (i, j) for the cost matrix
    pairs = [ (i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1) ]
    cost_map = dict(zip(pairs, A_flat))
    
    # Adjacency sets for G and H for O(1) lookup
    adj_G = set(G_edges)
    adj_H = set(H_edges)
    
    # We need to find a permutation P of (1...N) such that 
    # the cost to make H isomorphic to G is minimized.
    # The cost for a permutation P is the sum of A_{P_i, P_j} 
    # for all pairs (i, j) where (edge in G) != (edge in H after mapping)
    # Wait, the problem says we modify H. 
    # If we map vertex i of G to vertex P_i of H:
    # An edge (i, j) in G must correspond to an edge (P_i, P_j) in H.
    # If (i, j) is an edge in G but (P_i, P_j) is NOT in H, we pay A_{P_i, P_j} to add it.
    # If (i, j) is NOT an edge in G but (P_i, P_j) IS in H, we pay A_{P_i, P_j} to remove it.
    
    # Pre-calculate all possible pairs (i, j) with 1 <= i < j <= N
    all_pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    
    # The permutation P maps vertex i of G to vertex P[i-1] of H.
    # For every pair (i, j) in G, we check if the edge status matches (P_i, P_j) in H.
    
    # To avoid loops, we use a generator expression inside min()
    # P is a tuple representing the permutation
    # We use a helper function to ensure the pair (P[i-1], P[j-1]) is sorted for the cost_map
    
    def get_cost(P):
        # P is 0-indexed, vertices are 1-indexed.
        # For each pair (i, j) with 1 <= i < j <= N:
        # Edge in G: (i, j) in adj_G
        # Edge in H: (sorted(P[i-1], P[j-1])) in adj_H
        # If they differ, add cost_map[sorted(P[i-1], P[j-1])]
        
        # We use a generator expression to sum the costs
        return sum(
            cost_map[tuple(sorted((P[i-1], P[j-1])))]
            for i, j in all_pairs
            if ((i, j) in adj_G) != (tuple(sorted((P[i-1], P[j-1]))) in adj_H)
        )

    # Try all permutations of (1...N)
    # Use map() or a generator expression to apply get_cost to all permutations
    ans = min(map(get_cost, permutations(range(1, N + 1))))
    
    print(ans)

if __name__ == "__main__":
    solve()