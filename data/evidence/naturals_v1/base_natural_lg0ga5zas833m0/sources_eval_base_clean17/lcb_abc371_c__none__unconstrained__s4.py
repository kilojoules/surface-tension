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
        
    # Cost matrix A
    # A is provided as a triangular matrix
    cost_a = [[0] * n for _ in range(n)]
    for i in range(n - 1):
        row = list(map(int, sys.stdin.readline().split()))
        for j in range(i + 1, n):
            val = row[j - (i + 1)]
            cost_a[i][j] = cost_a[j][i] = val

    # To make G and H isomorphic, we need a permutation P of vertices of H
    # such that we transform H into G' where G' is G mapped by P.
    # The cost is the sum of A_{i,j} for all edges (i,j) that differ between H and G'.
    # Wait, the problem says "make G and H isomorphic".
    # This means we modify H to H' such that H' is isomorphic to G.
    # H' is isomorphic to G if there exists a permutation P such that
    # edge (i, j) in G exists iff edge (P_i, P_j) in H' exists.
    # To minimize cost, we fix a permutation P and calculate the cost to make H'
    # exactly the graph where edge (P_i, P_j) exists iff edge (i, j) exists in G.
    # The cost for a fixed P is:
    # Sum_{1 <= i < j <= N} (A_{P_i, P_j} if (edge (i, j) in G != edge (P_i, P_j) in H))
    
    min_total_cost = float('inf')
    
    # Permutations of (0, 1, ..., N-1)
    # N is up to 8, so N! = 40320, which is feasible.
    for p in permutations(range(n)):
        current_cost = 0
        # We only need to iterate over pairs i < j in G
        for i in range(n):
            for j in range(i + 1, n):
                # The edge in G is (i, j). The corresponding edge in H is (p[i], p[j]).
                # If they differ, we pay the cost A_{p[i], p[j]}.
                if adj_g[i][j] != adj_h[p[i]][p[j]]:
                    current_cost += cost_a[p[i]][p[j]]
        
        if current_cost < min_total_cost:
            min_total_cost = current_cost
            
    print(min_total_cost)

if __name__ == "__main__":
    solve()