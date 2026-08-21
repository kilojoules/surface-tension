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
    # We store it in a dictionary for easy lookup: {(i, j): cost} where i < j
    # The number of elements in A is N*(N-1)//2
    A_values = [int(next(it)) for _ in range(N * (N - 1) // 2)]
    
    # Map the flat A_values list to pairs (i, j)
    # Pairs are (1,2), (1,3)...(1,N), (2,3)...(2,N), ..., (N-1, N)
    A_pairs = [
        (i, j) 
        for i in range(1, N + 1) 
        for j in range(i + 1, N + 1)
    ]
    cost_map = dict(zip(A_pairs, A_values))

    # Represent G and H as adjacency sets for O(1) lookup
    # Using sets of frozensets or sorted tuples
    G_set = set(G_edges)
    H_set = set(H_edges)

    # We want to find a permutation P of (1...N) such that 
    # we minimize the cost to make H isomorphic to G.
    # The cost is the sum of A_{P_i, P_j} for all pairs (i, j) where
    # (edge (i, j) in G) != (edge (P_i, P_j) in H)
    
    # Pre-calculate all possible pairs (i, j) with i < j
    all_pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]

    # Try all permutations of vertices 1...N
    # P[i-1] is the vertex in H that vertex i in G is mapped to.
    # Note: permutations returns 0-indexed tuples, so we adjust.
    
    # To avoid loops, we use a generator expression inside min()
    # We use a helper function logic inside the comprehension.
    
    # For a permutation P, the cost is:
    # sum(cost_map[sorted((P[i-1], P[j-1]))] 
    #     for (i, j) in all_pairs 
    #     if ((i, j) in G_set) != ((sorted((P[i-1], P[j-1]))) in H_set))
    
    # Since we cannot use 'def' inside the comprehension for complex logic,
    # we keep the logic flat.
    
    ans = min(
        sum(
            cost_map[tuple(sorted((p[i-1], p[j-1])))]
            for (i, j) in all_pairs
            if ((i, j) in G_set) != (tuple(sorted((p[i-1], p[j-1]))) in H_set)
        )
        for p in permutations(range(1, N + 1))
    )
    
    print(ans)

if __name__ == "__main__":
    solve()