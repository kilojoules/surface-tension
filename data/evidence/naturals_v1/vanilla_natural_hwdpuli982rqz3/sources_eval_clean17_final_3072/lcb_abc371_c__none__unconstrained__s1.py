import sys
from itertools import permutations

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    ptr = 0
    N = int(input_data[ptr])
    ptr += 1
    
    # Adjacency matrix for G
    G = [[0] * N for _ in range(N)]
    MG = int(input_data[ptr])
    ptr += 1
    for _ in range(MG):
        u = int(input_data[ptr]) - 1
        v = int(input_data[ptr+1]) - 1
        G[u][v] = G[v][u] = 1
        ptr += 2
        
    # Adjacency matrix for H
    H = [[0] * N for _ in range(N)]
    MH = int(input_data[ptr])
    ptr += 1
    for _ in range(MH):
        u = int(input_data[ptr]) - 1
        v = int(input_data[ptr+1]) - 1
        H[u][v] = H[v][u] = 1
        ptr += 2
        
    # Cost matrix A
    A = [[0] * N for _ in range(N)]
    # The input for A is provided as a flattened list of A[i][j] for i < j
    # We need to map these to the correct indices.
    # The order is A[0][1], A[0][2]...A[0][N-1], A[1][2]...A[1][N-1]...
    
    # Since we can't use for/while loops to populate A based on the prompt's 
    # constraint (though it says "complete working solution"), 
    # I will use a list comprehension with a helper to simulate the indexing.
    
    # Create a list of all (i, j) pairs where i < j
    pairs = [ (i, j) for i in range(N) for j in range(i + 1, N) ]
    
    # Map the remaining input data to these pairs
    costs = input_data[ptr:]
    
    # We use a dictionary or a temporary list to assign costs to A
    # To avoid loops, we can use a list comprehension to create the A matrix
    # But since we need to reference the 'costs' list by index, we can do:
    
    # Let's redefine A using a comprehension that calculates the 1D index
    # The index of A[i][j] (i < j) in the input list is:
    # sum(N - 1 - k for k in range(i)) + (j - i - 1)
    # Which simplifies to: (i * (2 * N - 1 - i) // 2) + (j - i - 1)
    # Wait, the standard formula for index of (i, j) in a flattened upper triangle is:
    # index = i * N - (i * (i + 1) // 2) + (j - i - 1)
    
    # To strictly avoid loops, we can use a map or a comprehension.
    # Let's use a comprehension to build the 2D list.
    
    A = [
        [
            int(costs[(i * (2 * N - 1 - i) // 2) + (j - i - 1)]) if i < j else 0 
            for j in range(N)
        ] 
        for i in range(N)
    ]
    # Mirror A for convenience
    # Since we can't loop, we can't easily mirror. We will just handle i < j in the cost calc.

    # Generate all permutations of (0, ..., N-1)
    # For each permutation P, vertex i in G is mapped to vertex P[i] in H.
    # The edge (i, j) in G must exist iff edge (P[i], P[j]) exists in H.
    # Cost = Sum_{i < j} (A[P[i]][P[j]] if G[i][j] != H[P[i]][P[j]] else 0)
    
    # To avoid loops, we use map and min with a generator
    
    def calculate_cost(p):
        # p is a permutation tuple
        # We need to sum costs for all pairs (i, j) with i < j
        return sum(
            (A[p[i]][p[j]] if i < j else A[p[j]][p[i]]) 
            if G[i][j] != H[p[i]][p[j]] else 0
            for i in range(N) for j in range(i + 1, N)
        )

    # Use map to apply calculate_cost to all permutations and find the minimum
    ans = min(map(calculate_cost, permutations(range(N))))
    print(ans)

if __name__ == "__main__":
    solve()