import sys
from itertools import permutations

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use an iterator to consume input data without explicit loops
    it = iter(input_data)
    
    N = int(next(it))
    
    # Read G edges
    MG = int(next(it))
    G_edges = [tuple(sorted((int(next(it)), int(next(it))))) for _ in range(MG)]
    
    # Read H edges
    MH = int(next(it))
    H_edges = [tuple(sorted((int(next(it)), int(next(it))))) for _ in range(MH)]
    
    # Read A matrix (costs)
    # A is given as A[1,2], A[1,3]... A[1,N], A[2,3]...
    # We store it in a dictionary for easy access: {(i, j): cost} where i < j
    A_values = [int(next(it)) for _ in range(N * (N - 1) // 2)]
    
    # Map the flat A_values list to pairs (i, j)
    # The pairs are (1,2), (1,3)...(1,N), (2,3)...(2,N), etc.
    A_map = {
        (i, j): A_values[(i - 1) * N - (i * (i - 1) // 2) + (j - i - 1)]
        for i in range(1, N)
        for j in range(i + 1, N + 1)
    }
    
    # Adjacency sets for fast lookup
    G_adj = set(G_edges)
    H_adj = set(H_edges)
    
    # All possible edges in a graph of size N
    all_edges = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    
    # For a given permutation P, vertex i in G maps to P[i-1] in H.
    # G has edge (i, j) iff H has edge (P[i-1], P[j-1]).
    # If they differ, we pay A[sorted(P[i-1], P[j-1])].
    
    # We iterate through all permutations P of (1...N)
    # P[i-1] is the image of vertex i.
    # The cost for a permutation P is the sum over all i < j of:
    # A[sorted(P[i-1], P[j-1])] if (edge (i,j) in G) != (edge (P[i-1], P[j-1]) in H)
    
    # To avoid loops, we use min() over a generator expression
    # and sum() within that to calculate the cost for each permutation.
    
    ans = min(
        sum(
            A_map[tuple(sorted((p[i-1], p[j-1])))]
            for i, j in all_edges
            if ((i, j) in G_adj) != (tuple(sorted((p[i-1], p[j-1]))) in H_adj)
        )
        for p in permutations(range(1, N + 1))
    )
    
    print(ans)

if __name__ == "__main__":
    solve()