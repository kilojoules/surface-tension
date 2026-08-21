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
    # The costs are provided in a flattened manner for i < j
    # We need to map them to a structure we can access by (i, j)
    costs_flat = list(map(int, input_data[ptr:]))
    
    # Create a cost lookup dictionary for pairs (i, j) where i < j
    # The input order is A_{1,2}, A_{1,3}... A_{1,N}, A_{2,3}... A_{2,N}...
    # We can pre-calculate the pairs in the exact order they appear in the input
    all_pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    cost_map = dict(zip(all_pairs, costs_flat))

    # Adjacency matrices for G and H (using sets of frozensets for fast lookup)
    # We use sets of pairs (min(u,v), max(u,v))
    G_adj = {tuple(sorted((u, v))) for u, v in G_edges_list}
    H_adj = {tuple(sorted((u, v))) for u, v in H_edges_list}

    # We need to find a permutation P of (1...N) that minimizes the cost
    # The cost for a permutation P is the sum of A_{P_i, P_j} for all pairs (i, j)
    # where the edge status of (i, j) in G differs from the edge status of (P_i, P_j) in H.
    
    # To avoid loops, we use a generator expression inside min()
    # P is a permutation of (1...N)
    # For every pair 1 <= i < j <= N:
    # If (edge (i,j) in G) != (edge (P_i, P_j) in H), add cost A_{P_i, P_j}
    
    # Note: P is 0-indexed in the permutation tuple, so we map it to 1-indexed vertices
    # Let P be a tuple where P[i-1] is the vertex in H corresponding to vertex i in G.
    
    ans = min(
        sum(
            cost_map[tuple(sorted((P[i-1], P[j-1])))]
            for i in range(1, N + 1)
            for j in range(i + 1, N + 1)
            if ((i, j) in G_adj) != (tuple(sorted((P[i-1], P[j-1]))) in H_adj)
        )
        for P in permutations(range(1, N + 1))
    )
    
    print(ans)

if __name__ == "__main__":
    solve()