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
    
    # Helper to get edge existence as a boolean matrix
    def get_adj_matrix(num_edges, iterator):
        adj = [[False] * N for _ in range(N)]
        for _ in range(num_edges):
            u, v = int(next(iterator)), int(next(iterator))
            adj[u-1][v-1] = adj[v-1][u-1] = True
        return adj

    # Read M_G and edges of G
    MG = int(next(it))
    adj_G = get_adj_matrix(MG, it)
    
    # Read M_H and edges of H
    MH = int(next(it))
    adj_H = get_adj_matrix(MH, it)
    
    # Read cost matrix A
    # The input for A is provided as a flattened list of A_{i,j} for 1 <= i < j <= N
    costs_flat = [int(x) for x in it]
    
    # Map (i, j) where i < j to the cost in the flattened list
    # Index calculation: for a pair (i, j) with 0 <= i < j < N, 
    # the number of pairs (x, y) with x < y and x < i is i * (N - 1 - (i-1)/2)? No.
    # The number of pairs (x, y) with x < y and x < i is:
    # For x=0: (N-1) pairs, For x=1: (N-2) pairs ... For x=i-1: (N-i) pairs.
    # Sum is i*N - (i*(i+1)//2).
    
    # To avoid complex math, we can just pre-calculate the cost matrix using a loop
    # but since loops are discouraged for logic, we use a list comprehension with a helper.
    
    # We need to map (i, j) to the cost. Let's use a dictionary or a 2D array.
    # Since we can't use for-loops to populate the 2D array, we use a trick:
    # The input A is given exactly in the order i=1..N-1, j=i+1..N.
    
    # We can reconstruct the cost matrix by iterating through all pairs and 
    # calculating the index in the flat list.
    # Index of (i, j) where 0 <= i < j < N is:
    # sum_{k=0}^{i-1} (N - 1 - k) = i*(N-1) - (i-1)*i//2
    
    cost_matrix = [[0] * N for _ in range(N)]
    # We use a list comprehension to trigger the population of the cost_matrix
    [cost_matrix.__setitem__(i, [0] + [0]*i + [costs_flat[i*(N-1) - (i*(i-1)//2) + (j-i-1)] for j in range(i+1, N)] + [0]*(N-1-j) if i < N-1 else []) 
     for i in range(N)]
    # The above is messy. Let's use a simpler way to build the cost matrix.
    
    # Correct index for A_{i,j} (0-indexed i, j):
    # i=0: j=1..N-1 (indices 0 to N-2)
    # i=1: j=2..N-1 (indices N-1 to N-1 + N-3)
    # The index of (i, j) is (i * N) - (i * (i + 1) // 2) + (j - i - 1)
    
    def get_cost(i, j):
        if i > j: i, j = j, i
        return costs_flat[(i * N) - (i * (i + 1) // 2) + (j - i - 1)]

    # Generate all permutations of (0, ..., N-1)
    # For each permutation P, vertex i in G is mapped to vertex P[i] in H.
    # Edge (i, j) exists in G iff edge (P[i], P[j]) exists in H.
    # If they differ, we pay the cost to flip the edge in H.
    
    # To minimize memory/loops, we use map and min with permutations.
    
    ans = min(
        sum(
            get_cost(p[i], p[j]) 
            for i in range(N) 
            for j in range(i + 1, N) 
            if adj_G[i][j] != adj_H[p[i]][p[j]]
        )
        for p in permutations(range(N))
    )
    
    print(ans)

if __name__ == "__main__":
    solve()