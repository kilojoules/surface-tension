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
    
    # Graph G adjacency matrix
    adj_G = [[0] * N for _ in range(N)]
    M_G = int(input_data[ptr])
    ptr += 1
    for _ in range(M_G):
        u = int(input_data[ptr]) - 1
        v = int(input_data[ptr+1]) - 1
        adj_G[u][v] = adj_G[v][u] = 1
        ptr += 2
        
    # Graph H adjacency matrix
    adj_H = [[0] * N for _ in range(N)]
    M_H = int(input_data[ptr])
    ptr += 1
    for _ in range(M_H):
        u = int(input_data[ptr]) - 1
        v = int(input_data[ptr+1]) - 1
        adj_H[u][v] = adj_H[v][u] = 1
        ptr += 2
        
    # Cost matrix A
    # A[i][j] is the cost to flip edge (i, j)
    # The input provides A_{1,2}, A_{1,3}... A_{N-1,N}
    cost_A = [[0] * N for _ in range(N)]
    for i in range(N - 1):
        for j in range(i + 1, N):
            cost_A[i][j] = cost_A[j][i] = int(input_data[ptr])
            ptr += 1
            
    # We need to find a permutation P of {0, ..., N-1} such that
    # the cost to make H isomorphic to G via P is minimized.
    # The cost for a permutation P is the sum of cost_A[P[i]][P[j]] 
    # for all pairs (i, j) where adj_G[i][j] != adj_H[P[i]][P[j]].
    
    # Pre-calculate all pairs to avoid nested loops in the permutation loop
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((i, j))
            
    min_total_cost = float('inf')
    
    # N is at most 8, so N! is 40,320. This is feasible.
    for P in permutations(range(N)):
        current_cost = 0
        for i, j in pairs:
            # Edge in G is (i, j). Corresponding edge in H is (P[i], P[j]).
            # If they differ, we must pay the cost to flip the edge in H.
            if adj_G[i][j] != adj_H[P[i]][P[j]]:
                current_cost += cost_A[P[i]][P[j]]
        
        if current_cost < min_total_cost:
            min_total_cost = current_cost
            
    print(min_total_cost)

if __name__ == "__main__":
    solve()