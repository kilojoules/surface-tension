import itertools
import sys

def solve():
    # Read all input data
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
    # We store it in a dictionary with sorted tuples as keys for easy access
    costs_flat = list(map(int, input_data[ptr:]))
    
    # Generate all pairs (i, j) with 1 <= i < j <= N in the order they appear in input
    all_pairs = [ (i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1) ]
    cost_map = {pair: cost for pair, cost in zip(all_pairs, costs_flat)}

    # Adjacency matrices for G and H
    # Using sets of frozensets for edge lookups
    G_adj = {frozenset([u, v]) for u, v in G_edges_list}
    H_adj = {frozenset([u, v]) for u, v in H_edges_list}

    # We need to find a permutation P of (1...N) that minimizes the cost
    # The cost for a permutation P is the sum of A_{P_i, P_j} for all pairs (i, j)
    # where (edge (i, j) in G) != (edge (P_i, P_j) in H)
    
    # Pre-calculate the cost for every pair (i, j) if the edge status changes
    # However, the cost A_{i,j} is associated with the vertices in H.
    # If we map vertex i of G to P_i of H, the edge (i, j) in G corresponds to (P_i, P_j) in H.
    # The cost to flip the edge (P_i, P_j) is A_{min(P_i, P_j), max(P_i, P_j)}.
    
    # To optimize, we can pre-calculate a cost matrix for all pairs of vertices in H
    # But since N is very small (<= 8), we can use a list comprehension inside min()
    
    # We iterate through all permutations of (1...N)
    # For each permutation P, we calculate the total cost.
    # P[i-1] is the vertex in H that vertex i in G is mapped to.
    
    # To avoid loops, we use a generator expression inside sum()
    # and a generator expression inside min()
    
    ans = min(
        sum(
            cost_map[(min(p1, p2), max(p1, p2))]
            for i, j in itertools.combinations(range(1, N + 1), 2)
            for p1, p2 in [(P[i-1], P[j-1])]
            if (frozenset([i, j]) in G_adj) != (frozenset([p1, p2]) in H_adj)
        )
        for P in itertools.permutations(range(1, N + 1))
    )
    
    print(ans)

if __name__ == "__main__":
    solve()