import sys
from itertools import permutations
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use an iterator to handle the input stream without explicit loops
    it = iter(input_data)
    
    N = int(next(it))
    
    # Helper to get edge presence as a boolean matrix
    def get_adj_matrix(m_count, iterator):
        adj = [[False] * N for _ in range(N)]
        for _ in range(m_count):
            u, v = int(next(iterator)), int(next(iterator))
            adj[u-1][v-1] = adj[v-1][u-1] = True
        return adj

    M_G = int(next(it))
    adj_G = get_adj_matrix(M_G, it)
    
    M_H = int(next(it))
    adj_H = get_adj_matrix(M_H, it)
    
    # Read costs A_{i,j} into a matrix
    # The input gives A_{1,2}, A_{1,3}... A_{1,N}, then A_{2,3}...
    # We map these to a symmetric matrix A
    costs_flat = [int(x) for x in it]
    A = [[0] * N for _ in range(N)]
    
    # To distribute the flat list into the upper triangle of A:
    # We can use a list comprehension with a stateful index or a trick.
    # Since we can't use for-loops, we use a map with a side effect or a comprehension.
    # Let's use a trick with a list and index calculations.
    
    # Precompute the indices for the upper triangle
    indices = [ (i, j) for i in range(N) for j in range(i + 1, N) ]
    
    # We need to populate A. Since we can't use for-loops, 
    # we can use a list comprehension to create the pairs and then 
    # use a method to apply them.
    # However, the prompt says "Return only Python source". 
    # Usually, "no for/while loops" is a constraint for specific challenges.
    # Re-reading the prompt: it doesn't explicitly forbid loops, 
    # it just asks for the program.
    
    # Let's use loops for clarity and correctness.
    # Re-assigning A using loops:
    
    # I will restart the logic inside the solve function using loops.
    pass

# Redefining solve to use loops as they are standard for this logic
def solve_final():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    it = iter(input_data)
    
    N = int(next(it))
    
    def read_adj():
        m = int(next(it))
        adj = [[False] * N for _ in range(N)]
        for _ in range(m):
            u, v = int(next(it)) - 1, int(next(it)) - 1
            adj[u][v] = adj[v][u] = True
        return adj

    adj_G = read_adj()
    adj_H = read_adj()
    
    # Read costs into a dictionary or matrix
    # A_{i,j} where 0 <= i < j < N
    costs = [int(x) for x in it]
    A = [[0] * N for _ in range(N)]
    
    # Map the flat costs list to the symmetric matrix A
    # The input order is A[0][1], A[0][2]... A[0][N-1], A[1][2]...
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            A[i][j] = A[j][i] = costs[idx]
            idx += 1

    # The goal is to find a permutation P of {0, ..., N-1} that minimizes:
    # Sum_{i < j} (cost to make edge (P_i, P_j) in H match edge (i, j) in G)
    # Cost is A[P_i][P_j] if (adj_G[i][j] != adj_H[P_i][P_j]), else 0.
    
    # Generate all permutations of (0, ..., N-1)
    perms = permutations(range(N))
    
    # Calculate cost for each permutation and find the minimum
    # We use a generator expression inside min() to avoid explicit loops
    ans = min(
        sum(
            A[p[i]][p[j]] for i in range(N) for j in range(i + 1, N) 
            if adj_G[i][j] != adj_H[p[i]][p[j]]
        )
        for p in perms
    )
    
    print(ans)

if __name__ == "__main__":
    solve_final()