import sys
from itertools import permutations

def solve():
    # Read N
    line = sys.stdin.readline()
    if not line:
        return
    n = int(line.strip())
    
    # Adjacency matrix for G
    g_adj = [[False] * n for _ in range(n)]
    mg = int(sys.stdin.readline().strip())
    for _ in range(mg):
        u, v = map(int, sys.stdin.readline().split())
        g_adj[u-1][v-1] = g_adj[v-1][u-1] = True
        
    # Adjacency matrix for H
    h_adj = [[False] * n for _ in range(n)]
    mh = int(sys.stdin.readline().strip())
    for _ in range(mh):
        u, v = map(int, sys.stdin.readline().split())
        h_adj[u-1][v-1] = h_adj[v-1][u-1] = True
        
    # Cost matrix A
    # A is provided as a triangular matrix
    a_costs = [[0] * n for _ in range(n)]
    for i in range(n - 1):
        row = list(map(int, sys.stdin.readline().split()))
        for j in range(i + 1, n):
            cost = row[j - (i + 1)]
            a_costs[i][j] = a_costs[j][i] = cost
            
    # We need to find a permutation P of (0, ..., N-1) such that
    # the cost to transform H into a graph isomorphic to G via P is minimized.
    # The cost is sum_{i < j} A_{P_i, P_j} if (i, j) in G != (P_i, P_j) in H.
    # Wait, the isomorphism definition says: 
    # (i, j) is an edge in G iff (P_i, P_j) is an edge in H.
    # If we want to make H isomorphic to G, we can modify H.
    # Let's say we fix a permutation P. We want H' to be isomorphic to G via P.
    # That means (i, j) is an edge in G iff (P_i, P_j) is an edge in H'.
    # The cost to change H to H' is sum_{u < v} A_{u, v} where (u, v) in H != (u, v) in H'.
    # Let u = P_i and v = P_j.
    # The cost is sum_{i < j} A_{P_i, P_j} if (i, j) in G != (P_i, P_j) in H.
    
    min_total_cost = float('inf')
    
    # Since N is small (up to 8), we can iterate through all N! permutations.
    # 8! = 40,320, which is feasible.
    for p in permutations(range(n)):
        current_cost = 0
        # Iterate over all pairs in G
        for i in range(n):
            for j in range(i + 1, n):
                # Edge status in G for (i, j)
                g_edge = g_adj[i][j]
                # Edge status in H for (P_i, P_j)
                h_edge = h_adj[p[i]][p[j]]
                
                if g_edge != h_edge:
                    current_cost += a_costs[p[i]][p[j]]
        
        if current_cost < min_total_cost:
            min_total_cost = current_cost
            
    print(min_total_cost)

if __name__ == "__main__":
    solve()