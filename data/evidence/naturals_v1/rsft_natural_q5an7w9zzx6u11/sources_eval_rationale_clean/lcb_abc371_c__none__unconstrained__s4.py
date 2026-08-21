import sys
from itertools import permutations

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    it = iter(input_data)
    N = int(next(it))
    
    # Graph G edges
    MG = int(next(it))
    G_edges = [tuple(sorted((int(next(it)), int(next(it))))) for _ in range(MG)]
    
    # Graph H edges
    MH = int(next(it))
    H_edges = [tuple(sorted((int(next(it)), int(next(it))))) for _ in range(MH)]
    
    # Cost matrix A
    # A[i][j] will store cost for pair (i+1, j+1)
    # The input provides A_{1,2}, A_{1,3}... A_{N-1,N}
    # We flatten the remaining input and map it to pairs
    A_flat = [int(x) for x in it]
    
    # Pre-calculate adjacency matrices for G and H
    # Using sets for O(1) lookup
    G_set = set(G_edges)
    H_set = set(H_edges)
    
    # Map the flat A list to a dictionary for easy access: {(i, j): cost}
    # There are N*(N-1)//2 pairs.
    pairs = [ (i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1) ]
    A_map = dict(zip(pairs, A_flat))
    
    # We need to find a permutation P of (1...N) such that 
    # transforming H to match G under P is minimized.
    # The cost to make H isomorphic to G under permutation P is:
    # Sum_{1 <= i < j <= N} A_{P_i, P_j} if (edge(i,j) in G != edge(P_i, P_j) in H)
    
    # To avoid loops, we use a generator expression inside min()
    # P is a permutation of (1, ..., N)
    # For every pair (i, j) with 1 <= i < j <= N:
    # We check if G has an edge (i, j) and if H has an edge (P[i-1], P[j-1])
    # If they differ, we add A_{sorted(P[i-1], P[j-1])}
    
    # Pre-calculate all pairs (i, j) for G
    G_pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    
    def calculate_cost(P):
        # P is the permutation tuple
        # For each pair (i, j) in G, check if the corresponding pair in H matches
        # The cost is based on the vertices in H: P[i-1] and P[j-1]
        return sum(
            A_map[tuple(sorted((P[i-1], P[j-1]))))]
            for i, j in G_pairs
            if ( (i, j) in G_set ) != (tuple(sorted((P[i-1], P[j-1 importer]))) in H_set)
        )
    
    # The above logic has a small typo in the lambda, let's refine it into a clean expression.
    # We want to minimize:
    # sum(A[sorted(P[i], P[j])] for i < j if G(i,j) != H(P[i], P[j]))
    
    # Corrected logic using a nested generator:
    ans = min(
        sum(
            A_map[tuple(sorted((p[i], p[j importer])))] # This is still slightly wrong
            for i in range(N) for j in range(i + 1, N)
        ) # Wait, the logic needs to be:
        for p in permutations(range(1, N + 1))
    )
    # Let's rewrite the final answer calculation clearly.
    
def main():
    input_data = sys.stdin.read().split()
    if not input_data: return
    it = iter(input_data)
    N = int(next(it))
    MG = int(next(it))
    G_set = set(tuple(sorted((int(next(it)), int(next(it))))) for _ in range(MG))
    MH = int(next(it))
    H_set = set(tuple(sorted((int(next(it)), int(next(it))))) for _ in range(MH))
    A_flat = [int(x) for x in it]
    pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    A_map = dict(zip(pairs, A_flat))
    
    # G_pairs are the indices we iterate over
    G_pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    
    # The cost for a permutation P is the sum of A_{P_i, P_j} 
    # for all pairs (i, j) where the edge existence differs between G and H.
    # P is a tuple where P[i-1] is the vertex in H corresponding to vertex i in G.
    
    print(min(
        sum(
            A_map[tuple(sorted((p[i-1], p[j-1]))))]
            for i, j in G_pairs
            if ((i, j) in G_set) != (tuple(sorted((p[i-1], p[j-1]))) in H_set)
        )
        for p in permutations(range(1, N + 1))
    ))

if __name__ == "__main__":
    main()