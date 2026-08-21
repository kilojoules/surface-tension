import sys
from itertools import permutations
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use an iterator to handle the input stream
    it = iter(input_data)
    
    # Read N
    N = int(next(it))
    
    # Helper to get edge presence as a bitmask or tuple
    # Since N is small (up to 8), we can represent the graph as an adjacency matrix
    # or a set of edges.
    
    # Read G edges
    MG = int(next(it))
    G_edges = [tuple(map(int, (next(it), next(it)))) for _ in range(MG)]
    
    # Read H edges
    MH = int(next(it))
    H_edges = [tuple(map(int, (next(it), next(it)))) for _ in range(MH)]
    
    # Read A costs into a 2D structure
    # A[i][j] where 1 <= i < j <= N
    # We'll store them in a dictionary or a nested list
    # Given the input format: A_{1,2}, A_{1,3}... A_{1,N}, A_{2,3}...
    costs_flat = [int(x) for x in it]
    
    # Map (i, j) with i < j to the cost in costs_flat
    # The index in costs_flat for (i, j) is:
    # For i=1: j-2
    # For i=2: (N-1) + (j-3)
    # General formula for 1-indexed i, j: 
    # index = sum_{k=1}^{i-1} (N-k) + (j-i-1)
    # Which simplifies to: (i-1)*N - i*(i-1)//2 + (j-i-1)
    
    # To make it easier, let's just pre-calculate the cost matrix
    cost_matrix = [[0] * (N + 1) for _ in range(N + 1)]
    idx = 0
    for i in range(1, N):
        for j in range(i + 1, N + 1):
            cost_matrix[i][j] = cost_matrix[j][i] = costs_flat[idx]
            idx += 1

    # Adjacency matrices for G and H
    adj_G = [[0] * (N + 1) for _ in range(N + 1)]
    for u, v in G_edges:
        adj_G[u][v] = adj_G[v][u] = 1
        
    adj_H = [[0] * (N + 1) for _ in range(N + 1)]
    for u, v in H_edges:
        adj_H[u][v] = adj_H[v][u] = 1

    # The goal is to find a permutation P of {1...N} that minimizes:
    # Sum_{1 <= i < j <= N} cost(P_i, P_j) if (edge(i, j) in G != edge(P_i, P_j) in H)
    
    # Generate all permutations of (1...N)
    # Note: We can fix P_1 = 1 to reduce search space if the graph was unlabeled, 
    # but G and H are specific graphs, and the cost matrix A is tied to specific vertices of H.
    # Wait, the isomorphism definition says: edge(i, j) in G iff edge(P_i, P_j) in H.
    # This means we are mapping vertex i of G to vertex P_i of H.
    
    nodes = list(range(1, N + 1))
    
    # We calculate the cost for a specific permutation P
    # P is a tuple where P[i-1] is the vertex in H that vertex i in G is mapped to.
    def calculate_cost(P):
        # Use a generator to sum costs for all pairs (i, j) with 1 <= i < j <= N
        return sum(
            cost_matrix[P[i-1]][P[j-1]] 
            for i in range(1, N) 
            for j in range(i + 1, N + 1) 
            if adj_G[i][j] != adj_H[P[i-1]][P[j-1]]
        )

    # Find the minimum cost across all permutations
    # map(calculate_cost, permutations(nodes)) is a generator
    # min() will consume it
    ans = min(map(calculate_cost, permutations(nodes)))
    print(ans)

if __name__ == "__main__":
    solve()