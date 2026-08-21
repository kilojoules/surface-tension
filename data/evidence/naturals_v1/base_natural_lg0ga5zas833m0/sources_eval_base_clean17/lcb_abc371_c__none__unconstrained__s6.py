import sys
from itertools import permutations

def solve():
    # Read N
    line = sys.stdin.readline()
    if not line:
        return
    n = int(line.strip())
    
    # Read Graph G
    m_g = int(sys.stdin.readline().strip())
    adj_g = [[False] * n for _ in range(n)]
    for _ in range(m_g):
        u, v = map(int, sys.stdin.readline().split())
        adj_g[u-1][v-1] = adj_g[v-1][u-1] = True
        
    # Read Graph H
    m_h = int(sys.stdin.readline().strip())
    adj_h = [[False] * n for _ in range(n)]
    for _ in range(m_h):
        u, v = map(int, sys.stdin.readline().split())
        adj_h[u-1][v-1] = adj_h[v-1][u-1] = True
        
    # Read Costs A_{i,j}
    # The input format for costs is a bit tricky: it's a triangular matrix.
    costs = [[0] * n for _ in range(n)]
    for i in range(n - 1):
        row = list(map(int, sys.stdin.readline().split()))
        for idx, val in enumerate(row):
            # Row i contains A_{i+1, i+2}, A_{i+1, i+3}...
            # So row[idx] corresponds to A_{i+1, i+2+idx}
            j = i + 1 + idx
            costs[i][j] = costs[j][i] = val

    # The problem asks for the minimum cost to make H isomorphic to G.
    # This means we need to find a permutation P of {0, ..., N-1} such that
    # the cost of transforming H into a graph G' where G' is isomorphic to G
    # under permutation P is minimized.
    # Actually, the condition is: we want to find a permutation P such that
    # if we map vertex i in G to vertex P[i] in H, the cost to make the edges match is minimized.
    # Edge (i, j) in G exists <=> Edge (P[i], P[j]) in H exists.
    # If they don't match, we pay cost A_{P[i], P[j]}.
    
    min_total_cost = float('inf')
    
    # N is small (up to 8), so we can iterate through all N! permutations.
    # 8! = 40,320, which is feasible.
    for p in permutations(range(n)):
        current_cost = 0
        # We only need to check pairs (i, j) where i < j
        for i in range(n):
            for j in range(i + 1, n):
                # Edge in G between i and j
                g_edge = adj_g[i][j]
                # Edge in H between P[i] and P[j]
                h_edge = adj_h[p[i]][p[j]]
                
                # If the edges differ, we must perform an operation on H
                if g_edge != h_edge:
                    current_cost += costs[p[i]][p[j]]
        
        if current_cost < min_total_cost:
            min_total_cost = current_cost
            
    print(min_total_cost)

if __name__ == "__main__":
    solve()