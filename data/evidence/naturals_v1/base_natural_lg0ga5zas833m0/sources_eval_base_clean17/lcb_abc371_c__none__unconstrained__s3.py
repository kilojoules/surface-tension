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
        
    # Costs for toggling edges in H
    # cost_matrix[i][j] will store A_{i+1, j+1}
    cost_matrix = [[0] * n for _ in range(n)]
    for i in range(n - 1):
        row = list(map(int, sys.stdin.readline().split()))
        for j in range(i + 1, n):
            cost_matrix[i][j] = cost_matrix[j][i] = row[j - (i + 1)]

    # We need to find a permutation P of {0, ..., n-1} such that
    # the cost to transform H into a graph isomorphic to G via P is minimized.
    # The target graph H' is defined such that (P[i], P[j]) is an edge in H' 
    # if and only if (i, j) is an edge in G.
    # The cost is the sum of A_{P[i], P[j]} for all (i, j) where 
    # (edge exists in G) != (edge exists in H).
    
    min_total_cost = float('inf')
    
    # N is small (up to 8), so N! is at most 40320.
    # We can iterate through all permutations of vertices.
    for p in permutations(range(n)):
        current_cost = 0
        # We only need to check pairs i < j
        for i in range(n):
            for j in range(i + 1, n):
                # Edge in G between i and j
                edge_g = adj_g[i][j]
                # Edge in H between p[i] and p[j]
                edge_h = adj_h[p[i]][p[j]]
                
                if edge_g != edge_h:
                    current_cost += cost_matrix[p[i]][p[j]]
        
        if current_cost < min_total_cost:
            min_total_cost = current_cost
            
    print(min_total_cost)

if __name__ == "__main__":
    solve()