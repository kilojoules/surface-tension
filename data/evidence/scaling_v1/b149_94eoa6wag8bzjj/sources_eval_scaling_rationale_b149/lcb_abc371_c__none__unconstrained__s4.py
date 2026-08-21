import sys
from itertools import permutations

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use a pointer-like approach via a generator to consume input
    it = iter(input_data)
    
    N = int(next(it))
    
    # Graph G edges
    MG = int(next(it))
    G_edges = [tuple(sorted((int(next(it)), int(next(it))))) for _ in range(MG)]
    
    # Graph H edges
    MH = int(next(it))
    H_edges = [tuple(sorted((int(next(it)), int(next(it))))) for _ in range(MH)]
    
    # Cost matrix A
    # A[i][j] will store cost for pair (i+1, j+1)
    # The input provides A_{1,2}, A_{1,3}... A_{N-1,N}
    # We flatten the remaining input and map it to pairs
    A_flat = [int(x) for x in it]
    
    # Create a lookup for costs between any two vertices i, j (1-indexed)
    # We use a dictionary with sorted tuples as keys
    all_pairs = [tuple(sorted((i, j))) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    cost_map = dict(zip(all_pairs, A_flat))
    
    # Adjacency sets for O(1) lookup
    G_adj = set(G_edges)
    H_adj = set(H_edges)
    
    # We need to find a permutation P of (1...N) such that 
    # the cost to make H isomorphic to G is minimized.
    # The cost for a permutation P is the sum of A_{P_i, P_j} for all pairs (i, j)
    # where (edge exists in G between i,j) != (edge exists in H between P_i, P_j)
    
    # Generate all permutations of vertices 1...N
    perms = permutations(range(1, N + 1))
    
    # For a fixed permutation P, the total cost is:
    # Sum_{1 <= i < j <= N} cost_map(sorted(P_i, P_j)) 
    # IF (i, j) in G_adj XOR (P_i, P_j) in H_adj
    
    # To avoid loops, we use nested list comprehensions and min()
    # We pre-calculate all pairs (i, j) to avoid redundant work
    pairs_i_j = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    
    ans = min(
        sum(
            cost_map[tuple(sorted((p[i-1], p[j-1]))))]
            for i, j in pairs_i_j
            if ((i, j) in G_adj) != (tuple(sorted((p[i-1], p[j-1]))) in H_adj)
        )
        for p in perms
    )
    
    print(ans)

if __name__ == "__main__":
    solve()