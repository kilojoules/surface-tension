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
    # A is provided as a flattened upper triangle
    # A[i][j] for 1 <= i < j <= N
    # We'll store it in a dictionary for easy access: {(i, j): cost}
    # The input order is A_{1,2}, A_{1,3}... A_{1,N}, A_{2,3}...
    costs_flat = list(map(int, input_data[ptr:]))
    
    # Generate the pairs (i, j) in the order they appear in the input
    all_pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    cost_map = dict(zip(all_pairs, costs_flat))

    # Adjacency matrices for G and H (using sets of frozensets for fast lookup)
    # We use frozenset({u, v}) to represent an undirected edge
    G_adj = {frozenset(edge) for edge in G_edges_list}
    H_adj = {frozenset(edge) for edge in H_edges_list}

    # We need to find a permutation P of (1...N) that minimizes the cost.
    # The cost for a permutation P is the sum of A_{P_i, P_j} for all pairs (i, j)
    # where the edge status of (i, j) in G differs from the edge status of (P_i, P_j) in H.
    # Wait, the definition of isomorphism says:
    # edge (i, j) in G <=> edge (P_i, P_j) in H.
    # The cost is paid to change H. So if (i, j) is an edge in G, we need (P_i, P_j) to be an edge in H.
    # If it isn't, we pay A_{P_i, P_j}. If (i, j) is NOT an edge in G, we need (P_i, P_j) NOT to be an edge in H.
    # If it is, we pay A_{P_i, P_j}.
    
    # Pre-calculate all possible pairs (i, j) with i < j
    pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    
    # We want to minimize sum(cost_map[sorted_pair(P_i, P_j)]) 
    # for all (i, j) where (edge(i, j) in G) != (edge(P_i, P_j) in H)
    
    # To avoid loops, we use a generator expression inside min()
    # P is a permutation of (1, ..., N)
    # For each pair (i, j) in G, we check if the corresponding pair (P[i-1], P[j-1]) in H 
    # has the same edge status.
    
    # Since N is small (<= 8), we can iterate through all N! permutations.
    # We use a helper function to get the cost of a specific permutation.
    
    def get_cost(P):
        # P is a tuple (P1, P2, ..., PN)
        # We need to check all pairs (i, j) with 1 <= i < j <= N
        # The cost is A_{P_i, P_j} if (edge i,j in G) != (edge P_i, P_j in H)
        # Note: P_i is P[i-1]
        return sum(
            cost_map[tuple(sorted((P[i-1], P[j-1])))]
            for i in range(1, N + 1)
            for j in range(i + 1, N + 1)
            if (frozenset({i, j}) in G_adj) != (frozenset({P[i-1], P[j-1]}) in H_adj)
        )

    # Use permutations from itertools and map the get_cost function
    # Then find the minimum.
    ans = min(map(get_cost, permutations(range(1, N + 1))))
    print(ans)

if __name__ == "__main__":
    solve()