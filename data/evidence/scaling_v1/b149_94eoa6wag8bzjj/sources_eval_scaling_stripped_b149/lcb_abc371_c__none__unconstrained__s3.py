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
    G_edges = []
    for _ in range(MG):
        u = int(input_data[ptr])
        v = int(input_data[ptr+1])
        G_edges.append(tuple(sorted((u, v))))
        ptr += 2
        
    # Graph H edges
    MH = int(input_data[ptr])
    ptr += 1
    H_edges = []
    for _ in range(MH):
        u = int(input_data[ptr])
        v = int(input_data[ptr+1])
        H_edges.append(tuple(sorted((u, v))))
        ptr += 2
        
    # Cost matrix A
    # A is provided as A_{1,2}, A_{1,3}... A_{1,N}, A_{2,3}...
    # We map this to a dictionary for easy access
    costs_flat = list(map(int, input_data[ptr:]))
    
    # Pre-calculate the indices of the cost matrix
    # pairs is a list of all (i, j) where 1 <= i < j <= N
    pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    cost_map = {pairs[k]: costs_flat[k] for k in range(len(pairs))}

    # Adjacency sets for G and H for O(1) lookup
    adj_G = set(G_edges)
    adj_H = set(H_edges)

    # We need to find a permutation P of (1...N) that minimizes:
    # Sum_{1 <= i < j <= N} cost(P_i, P_j) if (edge(i,j) in G != edge(P_i, P_j) in H)
    # Note: The problem says G and H are isomorphic if there is a permutation P 
    # such that edge(i,j) exists in G iff edge(P_i, P_j) exists in H.
    # The cost is paid to change H to match G under permutation P.
    
    # Generate all permutations of 1...N
    all_perms = permutations(range(1, N + 1))
    
    # For a fixed permutation P, the cost is:
    # sum(cost(P_i, P_j) for i < j if (is_edge(i, j, G) != is_edge(P_i, P_j, H)))
    # However, the cost A_{i,j} is associated with the vertices of H.
    # If we map vertex i of G to vertex P_i of H, the edge (i, j) in G 
    # corresponds to edge (P_i, P_j) in H.
    # If one exists and the other doesn't, we pay A_{min(P_i, P_j), max(P_i, P_j)}.
    
    def calculate_cost(p):
        # p is a permutation where p[i-1] is the image of vertex i
        # We iterate over all pairs (i, j) with 1 <= i < j <= N
        # The corresponding vertices in H are p[i-1] and p[j-1]
        return sum(
            cost_map[tuple(sorted((p[i-1], p[j-1])))]
            for i in range(1, N + 1)
            for j in range(i + 1, N + 1)
            if ((i, j) in adj_G) != (tuple(sorted((p[i-1], p[j-1]))) in adj_H)
        )

    # Use min() with a generator expression to find the minimum cost across all permutations
    print(min(calculate_cost(p) for p in all_perms))

if __name__ == "__main__":
    solve()