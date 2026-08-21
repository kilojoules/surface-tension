import sys
from itertools import permutations

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
    # The costs are provided for pairs (i, j) where 1 <= i < j <= N
    # We store them in a dictionary for O(1) access
    costs_list = input_data[ptr:]
    
    # To map the flat list of costs to pairs (i, j), we generate all pairs
    all_pairs = [tuple(sorted((i, j))) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    cost_map = {pair: int(costs_list[idx]) for idx, pair in enumerate(all_pairs)}

    # Convert edge lists to sets for O(1) lookup
    set_G = set(G_edges)
    set_H = set(H_edges)

    # We need to find a permutation P of (1...N) that minimizes the cost.
    # The cost for a permutation P is the sum of A_{P_i, P_j} for all pairs (i, j)
    # where the edge status in G differs from the edge status in H.
    # Specifically: edge (i, j) in G XOR edge (P_i, P_j) in H.
    
    # Generate all permutations of 1...N
    all_perms = permutations(range(1, N + 1))
    
    # For each permutation, calculate the total cost
    # We use a generator expression inside min()
    ans = min(
        sum(
            cost_map[tuple(sorted((p[i-1], p[j-1])))]
            for i in range(1, N + 1)
            for j in range(i + 1, N + 1)
            if (tuple(sorted((i, j))) in set_G) != (tuple(sorted((p[i-1], p[j-1]))) in set_H)
        )
        for p in all_perms
    )
    
    print(ans)

if __name__ == "__main__":
    solve()