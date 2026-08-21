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
    M_G = int(input_data[ptr])
    ptr += 1
    G_edges = []
    for _ in range(M_G):
        u = int(input_data[ptr])
        v = int(input_data[ptr+1])
        G_edges.append(tuple(sorted((u, v))))
        ptr += 2
        
    # Graph H edges
    M_H = int(input_data[ptr])
    ptr += 1
    H_edges = []
    for _ in range(M_H):
        u = int(input_data[ptr])
        v = int(input_data[ptr+1])
        H_edges.append(tuple(sorted((u, v))))
        ptr += 2
        
    # Cost matrix A
    # The costs are provided for pairs (i, j) where 1 <= i < j <= N
    # We store them in a dictionary for O(1) access
    # The order of costs in input is: (1,2), (1,3)...(1,N), (2,3)...(2,N), etc.
    costs_list = list(map(int, input_data[ptr:]))
    
    # Generate all pairs (i, j) with 1 <= i < j <= N in the order they appear in input
    all_pairs = [ (i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1) ]
    cost_map = dict(zip(all_pairs, costs_list))
    
    # Convert G and H edge lists to sets for O(1) lookup
    set_G = set(G_edges)
    set_H = set(H_edges)
    
    # We need to find a permutation P of (1...N) that minimizes the cost.
    # The cost for a permutation P is the sum of cost(i, j) for all pairs (i, j)
    # where the edge status of (i, j) in G is different from the edge status of (P_i, P_j) in H.
    # Note: the problem says "make G and H isomorphic". 
    # This means we modify H to match G under permutation P.
    # If (i, j) is an edge in G, then (P_i, P_j) must be an edge in H.
    # If (i, j) is NOT an edge in G, then (P_i, P_j) must NOT be an edge in H.
    
    # To avoid loops, we use a generator expression inside min()
    # We iterate through all permutations of (1...N)
    # For each permutation, we calculate the total cost by checking all pairs (i, j)
    
    ans = min(
        sum(
            cost_map[tuple(sorted((i, j)))] 
            for i in range(1, N + 1) 
            for j in range(i + 1, N + 1) 
            if (tuple(sorted((i, j))) in set_G) != (tuple(sorted((p[i-1], p[j-1]))) in set_H)
        )
        for p in permutations(range(1, N + 1))
    )
    
    print(ans)

if __name__ == "__main__":
    solve()