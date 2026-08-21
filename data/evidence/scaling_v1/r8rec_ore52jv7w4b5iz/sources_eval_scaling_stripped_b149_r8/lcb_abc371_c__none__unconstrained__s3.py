import sys
from itertools import permutations

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    ptr = 0
    N = int(input_data[ptr])
    ptr += 1
    
    # Graph G edges
    MG = int(input_data[ptr])
    ptr += 1
    G_edges_list = []
    for _ in range(MG):
        u = int(input_data[ptr])
        v = int(input_data[ptr+1])
        G_edges_list.append((u, v))
        ptr += 2
        
    # Graph H edges
    MH = int(input_data[ptr])
    ptr += 1
    H_edges_list = []
    for _ in range(MH):
        u = int(input_data[ptr])
        v = int(input_data[ptr+1])
        H_edges_list.append((u, v))
        ptr += 2
        
    # Cost matrix A
    # A is provided as A_{1,2}, A_{1,3}... A_{1,N}, A_{2,3}...
    # We store it in a dictionary for easy access: (i, j) where i < j
    costs_flat = list(map(int, input_data[ptr:]))
    
    # Pre-calculate the indices of the cost matrix
    # There are N*(N-1)//2 pairs.
    # We can map (i, j) to an index using a formula or a list.
    # For N=8, we can just build a dictionary.
    
    # To build the cost dictionary without loops, we use a comprehension
    # The input order is A[1,2], A[1,3]... A[1,N], A[2,3]... A[2,N], A[3,4]...
    # We generate all pairs (i, j) with 1 <= i < j <= N in that specific order.
    all_pairs = [ (i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1) ]
    cost_dict = dict(zip(all_pairs, costs_flat))

    # Adjacency matrices for G and H (using sets of frozensets for fast lookup)
    # G_adj: set of edges in G
    G_adj = {frozenset(edge) for edge in G_edges_list}
    # H_adj: set of edges in H
    H_adj = {frozenset(edge) for edge in H_edges_list}

    # We need to find a permutation P of (1...N) that minimizes:
    # Sum_{1 <= i < j <= N} cost(P_i, P_j) if (edge(i,j) in G != edge(P_i, P_j) in H)
    
    # Generate all permutations of (1...N)
    perms = permutations(range(1, N + 1))
    
    # For a fixed permutation P, the cost is the sum over all pairs (i, j)
    # where the existence of edge (i, j) in G differs from edge (P_i, P_j) in H.
    # Note: P is 0-indexed in the permutation tuple, so P[i-1] is the vertex mapping.
    
    # To avoid loops, we use a generator expression inside min()
    # We iterate over all permutations P.
    # For each P, we iterate over all pairs (i, j) with 1 <= i < j <= N.
    
    ans = min(
        sum(
            cost_dict[tuple(sorted((p[i-1], p[j-1])))]
            for i in range(1, N + 1)
            for j in range(i + 1, N + 1)
            if (frozenset((i, j)) in G_adj) != (frozenset((p[i-1], p[j-1])) in H_adj)
        )
        for p in perms
    )
    
    print(ans)

if __name__ == "__main__":
    solve()