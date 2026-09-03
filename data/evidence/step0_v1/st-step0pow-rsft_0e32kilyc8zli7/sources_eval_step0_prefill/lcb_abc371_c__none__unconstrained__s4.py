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
    G = [[0] * N for _ in range(N)]
    MG = int(input_data[ptr])
    ptr += 1
    for _ in range(MG):
        u = int(input_data[ptr]) - 1
        v = int(input_data[ptr+1]) - 1
        G[u][v] = G[v][u] = 1
        ptr += 2
        
    # Graph H adjacency matrix
    H = [[0] * N for _ in range(N)]
    MH = int(input_data[ptr])
    ptr += 1
    for _ in range(MH):
        a = int(input_data[ptr]) - 1
        b = int(input_data[ptr+1]) - 1
        H[a][b] = H[b][a] = 1
        ptr += 2
        
    # Cost matrix A
    # A[i][j] is the cost to flip edge (i, j)
    A = [[0] * N for _ in range(N)]
    for i in range(N - 1):
        for j in range(i + 1, N):
            val = int(input_data[ptr])
            A[i][j] = A[j][i] = val
            ptr += 1

    # We want to find a permutation P of {0, ..., N-1} such that
    # the cost to make H isomorphic to G via P is minimized.
    # The cost for a permutation P is the sum over i < j of:
    # A[P[i]][P[j]] if (G[i][j] != H[P[i]][P[j]]) else 0.
    # Wait, the definition of isomorphism says:
    # edge (i, j) in G <=> edge (P[i], P[j]) in H.
    # So for a fixed permutation P, we need to change H such that
    # for all i < j, H[P[i]][P[j]] becomes equal to G[i][j].
    # The cost to change H[P[i]][P[j]] is A[P[i]][P[j]].
    
    min_total_cost = float('inf')
    
    # N is at most 8, so N! is 40320, which is feasible.
    for P in permutations(range(N)):
        current_cost = 0
        # Iterate over all pairs in G
        for i in range(N):
            for j in range(i + 1, N):
                # We need H[P[i]][P[j]] to be equal to G[i][j]
                if H[P[i]][P[j]] != G[i][j]:
                    current_cost += A[P[i]][P[j]]
        
        if current_cost < min_total_cost:
            min_total_cost = current_cost
            
    print(min_total_cost)

if __name__ == "__main__":
    solve()