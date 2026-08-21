import sys
from itertools import permutations

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use a pointer-like approach via an iterator to consume input without indices
    it = iter(input_data)
    
    N = int(next(it))
    
    # Graph G adjacency matrix
    MG = int(next(it))
    G_edges = [tuple(map(int, (next(it), next(it)))) for _ in range(MG)]
    # Use a set for O(1) edge lookup
    G_set = {tuple(sorted(e)) for e in G_edges}
    
    # Graph H adjacency matrix
    MH = int(next(it))
    H_edges = [tuple(map(int, (next(it), next(it)))) for _ in range(MH)]
    H_set = {tuple(sorted(e)) for e in H_edges}
    
    # Cost matrix A
    # A[i][j] will store cost for edge (i+1, j+1)
    # We read the costs and organize them into a dictionary for easy access
    def read_costs(n, current_it):
        # There are N-1 rows, then N-2, ..., 1
        # We can just read all remaining and map them to pairs (i, j)
        all_costs = [int(x) for x in current_it]
        
        # Generate all pairs (i, j) with 1 <= i < j <= N
        pairs = [(i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1)]
        return dict(zip(pairs, all_costs))

    A = read_costs(N, it)

    # We need to find a permutation P of (1...N) such that 
    # the cost to make H isomorphic to G is minimized.
    # The cost for a permutation P is the sum of A_{P_i, P_j} 
    # for all pairs (i, j) where (edge in G) != (edge in H after mapping)
    
    # Pre-calculate all pairs (i, j) for the cost summation
    all_pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    
    # The cost to transform H to G given permutation P:
    # For every pair (i, j) in G:
    # If (i, j) is an edge in G, we need (P[i], P[j]) to be an edge in H.
    # If (i, j) is NOT an edge in G, we need (P[i], P[j]) to NOT be an edge in H.
    # If the condition is not met, we pay A_{P[i], P[j]}.
    
    def calculate_cost(P):
        # P is a tuple representing the mapping: vertex i -> P[i-1]
        # We iterate over all pairs (i, j) and check if the edge status matches
        return sum(
            A[tuple(sorted((P[i-1], P[j-1]))))]
            for i, j in all_pairs
            if ((i, j) in G_set) != (tuple(sorted((P[i-1], P[j-1]))) in H_set)
        )

    # Try all permutations of (1...N) and find the minimum cost
    # Using a generator expression inside min()
    ans = min(calculate_cost(P) for P in permutations(range(1, N + 1)))
    
    print(ans)

if __name__ == "__main__":
    solve()