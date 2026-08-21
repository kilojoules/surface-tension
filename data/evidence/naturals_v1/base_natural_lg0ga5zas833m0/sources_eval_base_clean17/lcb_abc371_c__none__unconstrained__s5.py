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
    m_g = int(sys.stdin.readline().strip())
    for _ in range(m_g):
        u, v = map(int, sys.stdin.readline().split())
        adj_g[u-1][v-1] = adj_g[v-1][u-1] = True
        
    # Adjacency matrix for H
    adj_h = [[False] * n for _ in range(n)]
    m_h = int(sys.stdin.readline().strip())
    for _ in range(m_h):
        u, v = map(int, sys.stdin.readline().split())
        adj_h[u-1][v-1] = adj_h[v-1][u-1] = True
        
    # Costs matrix A
    # The input format for A is a bit unusual (triangular)
    costs = [[0] * n for _ in range(n)]
    for i in range(n - 1):
        row = list(map(int, sys.stdin.readline().split()))
        for j in range(i + 1, n):
            # row[j - (i + 1)] corresponds to A_{i+1, j+1}
            val = row[j - (i + 1)]
            costs[i][j] = costs[j][i] = val

    # Since N is very small (up to 8), we can iterate through all permutations P
    # of (0, 1, ..., N-1). P[i] will be the vertex in H that corresponds to 
    # vertex i in G.
    # The condition for isomorphism: edge (i, j) in G exists iff edge (P[i], P[j]) in H exists.
    # To make H isomorphic to G under permutation P, we need to flip edges in H
    # at positions (P[i], P[j]) where the state doesn't match G's (i, j).
    
    min_total_cost = float('inf')
    
    # Generate all permutations of 0...N-1
    for p in permutations(range(n)):
        current_cost = 0
        # Check all pairs (i, j) in G
        for i in range(n):
            for j in range(i + 1, n):
                # The edge in G is (i, j). The corresponding edge in H is (p[i], p[j]).
                # If their existence differs, we must pay to flip the edge in H.
                if adj_g[i][j] != adj_h[p[i]][p[j]]:
                    current_cost += costs[p[i]][p[j]]
        
        if current_cost < min_total_cost:
            min_total_cost = current_cost
            
    print(min_total_cost)

if __name__ == "__main__":
    solve()