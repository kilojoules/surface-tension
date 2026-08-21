import sys
from itertools import permutations

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = map(int, sys.stdin.read().split())
    gen = iter(input_data)
    
    # N: Number of vertices
    N = next(gen)
    
    # Graph G edges
    Mg = next(gen)
    g_edges = []
    for _ in range(Mg):
        g_edges.append(tuple(sorted((next(gen), next(gen)))))
    
    # Graph H edges
    Mh = next(gen)
    h_edges = []
    for _ in range(Mh):
        h_edges.append(tuple(sorted((next(gen), next(gen)))))
        
    # Cost matrix A_{i,j}
    # We store it in a dictionary for easy access: {(i, j): cost} where i < j
    # The input provides A_{1,2}, A_{1,3}... A_{N-1,N}
    costs_list = [next(gen) for _ in range(N * (N - 1) // 2)]
    
    # Map the flat cost list to pairs (i, j)
    # Pairs are (1,2), (1,3)...(1,N), (2,3)...(2,N), etc.
    cost_map = {
        (i, j): costs_list[(i-1)*N - (i*(i+1)//2) + (j-i-1)] 
        for i in range(1, N) for j in range(i + 1, N + 1)
    }
    # Correction on index logic for cost_map:
    # The input sequence is A_{1,2}, A_{1,3}... A_{1,N}, A_{2,3}...
    # Let's redefine cost_map using a simpler comprehension:
    all_pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    cost_dict = dict(zip(all_pairs, costs_list))

    # Adjacency matrices for G and H
    adj_g = [[False] * (N + 1) for _ in range(N + 1)]
    for u, v in g_edges:
        adj_g[u][v] = adj_g[v][u] = True
        
    adj_h = [[False] * (N + 1) for _ in range(N + 1)]
    for u, v in h_edges:
        adj_h[u][v] = adj_h[v][u] = True

    # We need to find a permutation P of (1...N) such that 
    # we minimize the cost of changing H to match G under mapping P.
    # Edge (i, j) in G exists iff edge (P[i], P[j]) in H exists.
    # If they differ, we pay cost A_{P[i], P[j]}.
    
    # Generate all permutations of vertices 1...N
    perms = permutations(range(1, N + 1))
    
    # For a given permutation P, the cost is the sum over all pairs (i, j)
    # where (i, j) is an edge in G XOR (P[i], P[j]) is an edge in H.
    # Note: P is 0-indexed in the permutation tuple, so vertex i is P[i-1].
    
    # To avoid loops, we use a generator expression inside min()
    # We pre-calculate all pairs (i, j) with i < j to iterate over.
    pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    
    # The cost for a specific permutation p:
    # p is a tuple where p[i-1] is the vertex in H corresponding to vertex i in G.
    # For pair (i, j) in G, the corresponding pair in H is (p[i-1], p[j-1]).
    # We need the cost A_{min(p[i-1], p[j-1]), max(p[i-1], p[j-1])}.
    
    ans = min(
        sum(
            cost_dict[tuple(sorted((p[i-1], p[j-1])))]
            for i, j in pairs
            if adj_g[i][j] != adj_h[p[i-1]][p[j-1]]
        )
        for p in perms
    )
    
    print(ans)

if __name__ == "__main__":
    solve()