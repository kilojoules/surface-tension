import sys
from itertools import permutations

def solve():
    # Read N
    line = sys.stdin.readline()
    if not line:
        return
    n = int(line.strip())
    
    # Adjacency matrix for G
    adj_g = [[False] * n for _ in range(n)]
    mg_line = sys.stdin.readline()
    if not mg_line:
        return
    mg = int(mg_line.strip())
    for _ in range(mg):
        u, v = map(int, sys.stdin.readline().split())
        adj_g[u-1][v-1] = adj_g[v-1][u-1] = True
        
    # Adjacency matrix for H
    adj_h = [[False] * n for _ in range(n)]
    mh_line = sys.stdin.readline()
    if not mh_line:
        return
    mh = int(mh_line.strip())
    for _ in range(mh):
        u, v = map(int, sys.stdin.readline().split())
        adj_h[u-1][v-1] = adj_h[v-1][u-1] = True
        
    # Cost matrix A
    # A[i][j] corresponds to cost for edge (i+1, j+1)
    # Input format is A_{1,2}, A_{1,3}... A_{1,N} then A_{2,3}...
    cost_matrix = [[0] * n for _ in range(n)]
    for i in range(n - 1):
        row = list(map(int, sys.stdin.readline().split()))
        for idx, val in enumerate(row):
            j = i + 1 + idx
            cost_matrix[i][j] = cost_matrix[j][i] = val
            
    # We need to find a permutation P of (0, ..., N-1) such that
    # the cost to transform H into a graph isomorphic to G via P is minimized.
    # The cost for a specific P is:
    # Sum_{0 <= i < j < N} cost_matrix[P[i]][P[j]] if (adj_g[i][j] != adj_h[P[i]][P[j]])
    
    min_total_cost = float('inf')
    
    # N is small (up to 8), so we can iterate through all N! permutations.
    # 8! = 40,320
    for p in permutations(range(n)):
        current_cost = 0
        # Check all pairs (i, j) in G and compare with (p[i], p[j]) in H
        for i in range(n):
            pi = p[i]
            for j in range(i + 1, n):
                pj = p[j]
                # If the edge existence differs between G(i,j) and H(p[i], p[j])
                if adj_g[i][j] != adj_h[pi][pj]:
                    current_cost += cost_matrix[pi][pj]
            
            # Early exit if current_cost already exceeds min_total_cost
            if current_cost >= min_total_cost:
                break
        
        if current_cost < min_total_cost:
            min_total_cost = current_cost
            
    print(min_total_cost)

if __name__ == "__main__":
    solve()