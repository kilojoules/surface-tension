import sys
from itertools import permutations

def solve():
    # Read N
    line = sys.stdin.readline()
    if not line:
        return
    n = int(line.strip())
    
    # Read Graph G edges
    mg = int(sys.stdin.readline().strip())
    adj_g = [[False] * n for _ in range(n)]
    for _ in range(mg):
        u, v = map(int, sys.stdin.readline().split())
        adj_g[u-1][v-1] = adj_g[v-1][u-1] = True
        
    # Read Graph H edges
    mh = int(sys.stdin.readline().strip())
    adj_h = [[False] * n for _ in range(n)]
    for _ in range(mh):
        u, v = map(int, sys.stdin.readline().split())
        adj_h[u-1][v-1] = adj_h[v-1][u-1] = True
        
    # Read costs A_{i,j}
    # A is provided as a lower triangular matrix without the diagonal
    cost_matrix = [[0] * n for _ in range(n)]
    for i in range(n - 1):
        row = list(map(int, sys.stdin.readline().split()))
        for j in range(i + 1, n):
            val = row[j - (i + 1)]
            cost_matrix[i][j] = cost_matrix[j][i] = val
            
    # The goal is to find a permutation P of {0, ..., N-1} such that
    # the cost to transform H into a graph isomorphic to G under P is minimized.
    # Under permutation P, vertex i in G maps to vertex P[i] in H.
    # An edge (i, j) exists in G if and only if (P[i], P[j]) exists in the modified H.
    # The cost for a pair (i, j) is A_{P[i], P[j]} if (adj_g[i][j] != adj_h[P[i]][P[j]]).
    
    min_total_cost = float('inf')
    
    # N is at most 8, so N! is at most 40,320.
    # We can iterate through all permutations.
    for p in permutations(range(n)):
        current_cost = 0
        # We only need to check pairs i < j
        for i in range(n):
            pi = p[i]
            for j in range(i + 1, n):
                pj = p[j]
                # If the edge state in G (i, j) differs from H (pi, pj), we pay A_{pi, pj}
                if adj_g[i][j] != adj_h[pi][pj]:
                    current_cost += cost_matrix[pi][pj]
            
            # Optimization: break if current_cost already exceeds min_total_cost
            if current_cost >= min_total_cost:
                break
                
        if current_cost < min_total_cost:
            min_total_cost = current_cost
            
    print(min_total_cost)

if __name__ == "__main__":
    solve()