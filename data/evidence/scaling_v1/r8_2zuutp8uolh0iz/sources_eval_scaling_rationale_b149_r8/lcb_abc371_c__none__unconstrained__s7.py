import sys

def solve():
    # Read N
    line = sys.stdin.readline()
    if not line:
        return
    n = int(line.strip())

    # Read Graph G
    m_g = int(sys.stdin.readline().strip())
    adj_g = [[0] * n for _ in range(n)]
    for _ in range(m_g):
        u, v = map(int, sys.stdin.readline().split())
        adj_g[u-1][v-1] = adj_g[v-1][u-1] = 1

    # Read Graph H
    m_h = int(sys.stdin.readline().strip())
    adj_h = [[0] * n for _ in range(n)]
    for _ in range(m_h):
        u, v = map(int, sys.stdin.readline().split())
        adj_h[u-1][v-1] = adj_h[v-1][u-1] = 1

    # Read Costs A_{i,j}
    # The input format for A is A_{1,2}, A_{1,3}... A_{N-1,N}
    # We store them in a 2D array for easy access
    costs = [[0] * n for _ in range(n)]
    for i in range(n - 1):
        row_costs = list(map(int, sys.stdin.readline().split()))
        for idx, val in enumerate(row_costs):
            j = i + 1 + idx
            costs[i][j] = costs[j][i] = val

    # Generate all permutations of (0, ..., N-1) manually using recursion
    all_perms = []
    def generate_permutations(current, remaining):
        if not remaining:
            all_perms.append(list(current))
            return
        for i in range(len(remaining)):
            current.append(remaining[i])
            # Create a new list for remaining to avoid mutation issues
            next_remaining = remaining[:i] + remaining[i+1:]
            generate_permutations(current, next_remaining)
            current.pop()

    generate_permutations([], list(range(n)))

    min_total_cost = float('inf')

    # For each permutation P, we check the cost to make H isomorphic to G
    # G edge (i, j) exists iff H edge (P[i], P[j]) exists.
    # If they differ, we pay costs[P[i]][P[j]]
    for p in all_perms:
        current_cost = 0
        # We only need to check pairs (i, j) where i < j
        for i in range(n):
            for j in range(i + 1, n):
                # Edge in G between i and j
                g_edge = adj_g[i][j]
                # Corresponding edge in H between P[i] and P[j]
                h_edge = adj_h[p[i]][p[j]]
                
                if g_edge != h_edge:
                    current_cost += costs[p[i]][p[j]]
        
        if current_cost < min_total_cost:
            min_total_cost = current_cost

    print(min_total_cost)

if __name__ == "__main__":
    solve()