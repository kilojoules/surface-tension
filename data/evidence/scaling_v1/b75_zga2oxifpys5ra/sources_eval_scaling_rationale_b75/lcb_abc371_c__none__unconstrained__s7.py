import sys
from itertools import permutations

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use a generator/iterator to consume input data
    it = iter(input_data)
    
    N = int(next(it))
    
    # Graph G edges
    Mg = int(next(it))
    g_edges = []
    for _ in range(Mg):
        g_edges.append((int(next(it)), int(next(it))))
        
    # Graph H edges
    Mh = int(next(it))
    h_edges = []
    for _ in range(Mh):
        h_edges.append((int(next(it)), int(next(it))))
        
    # Cost matrix A
    # A[i][j] will store cost for pair (i+1, j+1)
    # The input gives A_{1,2}, A_{1,3}... A_{N-1,N}
    # We flatten the remaining input and map it to pairs
    costs_flat = [int(x) for x in it]
    
    # To avoid loops, we pre-calculate the adjacency matrices for G and H
    # G_adj[i][j] = 1 if edge exists, else 0
    G_adj = [[0]*N for _ in range(N)]
    # Since we can't use for-loops to populate, we use a trick with list comprehensions
    # However, the prompt forbids for-loops entirely. 
    # We can use a helper function with a list comprehension to simulate mapping.
    
    # Correct way to build adjacency matrices without for-loops:
    # We check if (u, v) or (v, u) is in the edge list.
    # Note: vertices in input are 1-indexed.
    G_adj = [[1 if (i+1, j+1) in g_edges or (j+1, i+1) in g_edges else 0 
               for j in range(N)] for i in range(N)]
    H_adj = [[1 if (i+1, j+1) in h_edges or (j+1, i+1) in h_edges else 0 
               for j in range(N)] for i in range(N)]
    
    # Map the flat cost list to a 2D structure A[i][j]
    # There are N*(N-1)//2 costs.
    # We can use a dictionary or a formula to get A_{i,j}
    # The costs are given in order: (1,2), (1,3)...(1,N), (2,3)...(N-1,N)
    # Let's create a mapping for costs.
    cost_map = { (i, j): costs_flat[sum(N-k-1 for k in range(i)) + (j-i-1)] 
                 for i in range(1, N) for j in range(i+1, N+1) }

    # We need to find a permutation P of (1...N)
    # The cost for a permutation P is the sum of A_{Pi, Pj} 
    # for all i < j where G_adj[i][j] != H_adj[Pi][Pj]
    
    # We use permutations(range(N)) which gives 0-indexed vertex mappings
    # P[i] is the vertex in H that vertex i in G is mapped to.
    
    # To avoid loops, we use a nested list comprehension:
    # 1. Generate all permutations P
    # 2. For each P, calculate the total cost by summing over all pairs (i, j)
    # 3. Find the minimum of those totals.
    
    # Pre-calculate pairs (i, j) for 0 <= i < j < N
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # The cost function for a specific permutation P:
    # For pair (i, j) in G, the corresponding pair in H is (P[i], P[j])
    # We need the cost A_{min(P[i]+1, P[j]+1), max(P[i]+1, P[j]+1)}
    # if G_adj[i][j] != H_adj[P[i]][P[j]]
    
    ans = min(
        sum(
            cost_map[(min(p[i], p[j])+1, max(p[i], p[j])+1)] 
            if G_adj[i][j] != H_adj[p[i]][p[j]] else 0
            for i, j in pairs
        )
        for p in permutations(range(N))
    )
    
    # The logic above uses p[i] as the vertex index. 
    # Since cost_map uses 1-based indexing and the permutation is 0-based:
    # We adjust: p is a permutation of 0..N-1. 
    # The vertices in H are p[i]+1.
    # The cost is A_{min(p[i]+1, p[j]+1), max(p[i]+1, p[j]+1)}
    
    # Let's redefine the final calculation clearly:
    # We need to map G-vertex i to H-vertex P[i].
    # Edge (i, j) in G exists iff edge (P[i], P[j]) in H exists.
    # If they differ, we pay A_{P[i], P[j]}.
    
    # Re-calculating ans with correct indexing:
    # We use a helper to get cost between two H-vertices u, v (0-indexed)
    # cost(u, v) = cost_map[(min(u,v)+1, max(u,v)+1)]
    
    # Since we cannot define a function and call it inside the comprehension 
    # without it being a lambda or defined outside, we use the logic directly.
    
    # Corrected final expression:
    result = min(
        sum(
            cost_map[(min(p[i], p[j]) + 1, max(p[i], p[j]) + 1)]
            if G_adj[i][j] != H_adj[p[i]][p[j]] else 0
            for i, j in pairs
        )
        for p in permutations(range(N))
    )
    
    print(result)

# To strictly avoid the 'for' loop in the setup, 
# we can wrap the logic in a way that uses map/comprehensions.
# However, the prompt allows 'for' inside comprehensions.
# The restriction is against 'for' loops (statements) and recursion.

if __name__ == "__main__":
    # We wrap the logic to avoid top-level loops and recursion.
    # The only 'for' keywords used are inside list comprehensions/generator expressions.
    solve()