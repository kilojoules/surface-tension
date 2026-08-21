import sys
from itertools import permutations

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use a pointer to track current position in input_data
    ptr = 0
    
    # N: Number of vertices
    N = int(input_data[ptr])
    ptr += 1
    
    # M_G: Number of edges in G
    M_G = int(input_data[ptr])
    ptr += 1
    
    # G_edges: Set of edges in G (stored as sorted tuples)
    G_edges = set()
    for _ in range(M_G):
        u, v = int(input_data[ptr]), int(input_data[ptr+1])
        G_edges.add(tuple(sorted((u, v))))
        ptr += 2
        
    # M_H: Number of edges in H
    M_H = int(input_data[ptr])
    ptr += 1
    
    # H_edges: Set of edges in H (stored as sorted tuples)
    H_edges = set()
    for _ in range(M_H):
        a, b = int(input_data[ptr]), int(input_data[ptr+1])
        H_edges.add(tuple(sorted((a, b))))
        ptr += 2
        
    # A: Cost matrix. We map (i, j) where i < j to the cost.
    # The input provides A_{1,2}, A_{1,3}... A_{1,N}, then A_{2,3}...
    # We build a dictionary for easy lookup.
    A_vals = input_data[ptr:]
    
    # Generate all pairs (i, j) with 1 <= i < j <= N in the order they appear in input
    all_pairs = [ (i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1) ]
    cost_map = {pair: int(val) for pair, val in zip(all_pairs, A_vals)}

    # We need to find a permutation P of (1...N) that minimizes the cost.
    # The cost for a permutation P is the sum of A_{P_i, P_j} for all pairs (i, j)
    # where the edge status in G (i, j) differs from the edge status in H (P_i, P_j).
    
    # Pre-calculate all possible pairs (i, j) with 1 <= i < j <= N
    vertex_pairs = all_pairs
    
    # We use a generator expression inside min() to find the minimum cost across all permutations.
    # For a permutation P, the cost is sum(cost_map[sorted(P_i, P_j)] if G_edge != H_edge)
    # Note: P is 0-indexed in itertools, so we map 0..N-1 to 1..N.
    
    ans = min(
        sum(
            cost_map[tuple(sorted((p[i-1], p[j-1])))]
            for i, j in vertex_pairs
            if ((i, j) in G_edges) != (tuple(sorted((p[i-1], p[j-1]))) in H_edges)
        )
        for p in permutations(range(1, N + 1))
    )
    
    print(ans)

if __name__ == "__main__":
    solve()