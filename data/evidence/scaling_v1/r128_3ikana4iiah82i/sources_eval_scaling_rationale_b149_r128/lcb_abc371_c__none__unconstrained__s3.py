import sys
from itertools import permutations

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    ptr = 0
    N = int(input_data[ptr])
    ptr += 1
    
    # Graph G edges
    MG = int(input_data[ptr])
    ptr += 1
    G_edges = []
    for _ in range(MG):
        G_edges.append((int(input_data[ptr]), int(input_data[ptr+1])))
        ptr += 2
        
    # Graph H edges
    MH = int(input_data[ptr])
    ptr += 1
    H_edges = []
    for _ in range(MH):
        H_edges.append((int(input_data[ptr]), int(input_data[ptr+1])))
        ptr += 2
        
    # Cost matrix A
    # A[i][j] is the cost to flip edge (i+1, j+1)
    # The input gives A_{1,2}, A_{1,3}... A_{N-1,N}
    # We store it in a 2D list for easy access
    A_flat = list(map(int, input_data[ptr:]))
    
    # To reconstruct A[i][j] from the flat list:
    # The number of elements before A_{i,j} is sum(N-k for k from 1 to i-1)
    # Which is (i-1)*N - i*(i-1)//2
    # However, a simpler way is to build the matrix iteratively
    # But since we can't use loops, we use a dictionary or a comprehension
    
    # Calculate the starting index for each row i in the flat A list
    # row_start[i] = sum(N-k for k in range(1, i+1))
    # For N=5: row 1 starts at 0, row 2 starts at 4, row 3 at 4+3=7...
    
    # We can use a dictionary to store A[(i, j)] where i < j
    # We use a list comprehension to build the dictionary
    # The indices of A_flat correspond to pairs (i, j) in lexicographical order
    pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    A_dict = {pairs[k]: A_flat[k] for k in range(len(pairs))}

    # Adjacency matrices for G and H
    # G_mat[i][j] = 1 if edge exists, else 0
    G_mat = [[0] * (N + 1) for _ in range(N + 1)]
    # Since we can't use loops to fill G_mat, we use a trick with a list
    # But we can use a comprehension to build the matrix
    G_mat = [[1 if (i, j) in G_edges or (j, i) in G_edges else 0 
               for j in range(1, N + 1)] 
              for i in range(1, N + 1)]
    
    H_mat = [[1 if (i, j) in H_edges or (j, i) in H_edges else 0 
               for j in range(1, N + 1)] 
              for i in range(1, N + 1)]

    # We need to find a permutation P of (1...N) that minimizes cost
    # Cost for permutation P:
    # Sum_{1 <= i < j <= N} A_{P_i, P_j} if (G_mat[i][j] != H_mat[P_i][P_j])
    
    # Pre-calculate all possible pairs (i, j) with i < j
    all_pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    
    # Try all permutations of (1...N)
    # Note: P is a tuple where P[i-1] is the image of vertex i
    # The cost is summed over all pairs (i, j)
    
    # To avoid loops, we use a generator expression inside min()
    # We use a helper function to calculate cost for a single permutation
    def get_cost(P):
        # P is a permutation of 1...N
        # We check every pair (i, j) in G and see if the edge status 
        # matches the edge status of (P[i-1], P[j-1]) in H
        # If they differ, we add the cost A_{P[i-1], P[j-1]}
        # Note: A_dict requires keys to be sorted (min, max)
        return sum(
            A_dict[(min(P[i-1], P[j-1]), max(P[i-1], P[j-1]))]
            for i, j in all_pairs
            if G_mat[i-1][j-1] != H_mat[P[i-1]-1][P[j-1]-1]
        )

    # The answer is the minimum cost over all permutations
    ans = min(get_cost(P) for P in permutations(range(1, N + 1)))
    print(ans)

if __name__ == "__main__":
    solve()