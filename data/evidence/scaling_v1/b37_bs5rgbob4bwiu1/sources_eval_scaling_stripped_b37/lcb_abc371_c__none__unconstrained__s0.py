import sys
from itertools import permutations

def solve():
    # Read all input at once and split into a list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    ptr = 0
    N = int(input_data[ptr])
    ptr += 1
    
    # Helper to get edges into a set of frozensets for easy lookup
    def read_edges(count, current_ptr, data):
        edges = set()
        for _ in range(count):
            u = int(data[current_ptr])
            v = int(data[current_ptr + 1])
            edges.add(frozenset([u, v]))
            current_ptr += 2
        return edges, current_ptr

    M_G = int(input_data[ptr])
    ptr += 1
    edges_G, ptr = read_edges(M_G, ptr, input_data)
    
    M_H = int(input_data[ptr])
    ptr += 1
    edges_H, ptr = read_edges(M_H, ptr, input_data)
    
    # Read A_{i,j} matrix
    # A is given as a triangular matrix. We map (i, j) where i < j to the cost.
    # We can store this in a dictionary where keys are frozensets {i, j}.
    A = {}
    # The number of A values is N*(N-1)//2
    # They are given in order: (1,2), (1,3)...(1,N), (2,3)...(2,N), ..., (N-1,N)
    # We can use a comprehension to generate the pairs and zip them with the remaining input.
    pairs = [frozenset([i, j]) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    costs = map(int, input_data[ptr:])
    A = dict(zip(pairs, costs))
    
    # We want to find a permutation P of (1...N) such that the cost of transforming
    # H to be isomorphic to G via P is minimized.
    # The cost for a permutation P is the sum of A_{P_i, P_j} for all pairs (i, j)
    # where the edge status in G(i, j) differs from the edge status in H(P_i, P_j).
    # Wait, the definition of isomorphism says: 
    # edge (i, j) in G exists iff edge (P_i, P_j) in H exists.
    # So for every pair 1 <= i < j <= N:
    # If (edge (i, j) in G) != (edge (P_i, j) in H), we pay A_{P_i, P_j}.
    
    # To optimize, we pre-calculate the edge existence in G and H.
    # Since N is small (<= 8), we can iterate through all N! permutations.
    
    # We use a helper function to calculate cost for a specific permutation P.
    # P is a tuple where P[i-1] is the image of vertex i.
    def get_cost(P):
        # P maps vertex i+1 to P[i]
        # For all 0 <= i < j < N:
        # Check if edge {i+1, j+1} in G has same status as {P[i], P[j]} in H.
        # If different, add A[{P[i], P[j]}]
        
        # We can use a generator expression inside sum()
        return sum(
            A[frozenset([P[i], P[j]])]
            for i in range(N)
            for j in range(i + 1, N)
            if (frozenset([i + 1, j + 1]) in edges_G) != (frozenset([P[i], P[j]]) in edges_H)
        )

    # Generate all permutations of (1, 2, ..., N)
    perms = permutations(range(1, N + 1))
    
    # Find the minimum cost across all permutations
    ans = min(get_cost(P) for P in perms)
    
    print(ans)

if __name__ == "__main__":
    solve()