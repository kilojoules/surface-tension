import sys
from itertools import permutations

def solve():
    # Read N
    line = sys.stdin.readline()
    if not line:
        return
    n = int(line.strip())
    
    # Graph G adjacency matrix
    g_adj = [[False] * n for _ in range(n)]
    m_g = int(sys.stdin.readline().strip())
    for _ in range(m_g):
        u, v = map(int, sys.stdin.readline().split())
        g_adj[u-1][v-1] = g_adj[v-1][u-1] = True
        
    # Graph H adjacency matrix
    h_adj = [[False] * n for _ in range(n)]
    m_h = int(sys.stdin.readline().strip())
    for _ in range(m_h):
        u, v = map(int, sys.stdin.readline().split())
        h_adj[u-1][v-1] = h_adj[v-1][u-1] = True
        
    # Cost matrix A
    # A is provided in a flattened format: A[0,1], A[0,2]... A[0,n-1], A[1,2]...
    a_costs = [[0] * n for _ in range(n)]
    for i in range(n - 1):
        row = list(map(int, sys.stdin.readline().split()))
        for j in range(i + 1, n):
            # The row contains costs for (i, i+1), (i, i+2)... (i, n-1)
            # row[0] corresponds to A[i, i+1], row[1] to A[i, i+2], etc.
            # The index in the row is j - (i + 1)
            cost = row[j - (i + 1)]
            a_costs[i][j] = a_costs[j][i] = cost

    # We need to find a permutation P of vertices such that modifying H to match
    # G under P has minimum cost.
    # Specifically, we want to find a permutation P such that for all i < j:
    # if g_adj[i][j] != h_adj[P[i]][P[j]], we pay a_costs[P[i]][P[j]].
    
    min_total_cost = float('inf')
    
    # N is small (up to 8), so we can iterate through all N! permutations.
    # 8! = 40,320, which is feasible in Python.
    for p in permutations(range(n)):
        current_cost = 0
        # Check all pairs (i, j) in G
        for i in range(n):
            pi = p[i]
            for j in range(i + 1, n):
                pj = p[j]
                # Edge exists in G between i and j
                g_edge = g_adj[i][j]
                # Edge exists in H between P[i] and P[j]
                h_edge = h_adj[pi][pj]
                
                if g_edge != h_edge:
                    current_cost += a_costs[pi][pj]
        
        if current_cost < min_total_cost:
            min_total_cost = current_cost
            
    print(min_total_cost)

if __name__ == "__main__":
    solve()