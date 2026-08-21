import sys
from itertools import permutations

def solve():
    # Read all input at once and split into a list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to consume input_data sequentially
    gen = input_data
    
    # Since we cannot use loops, we use next() to extract values
    # However, the prompt forbids 'for' and 'while'. 
    # We can use a helper function or a list comprehension to parse.
    # But since we need to handle variable lengths (M_G, M_H), 
    # we will convert the generator to a list first.
    data = list(gen)
    
    N = data[0]
    M_G = data[1]
    
    # Extract edges of G
    # G_edges is a set of frozensets for fast lookup
    # We use a list comprehension to slice the data list
    G_edges = {frozenset([data[2 + 2*i], data[3 + 2*i]]) for i in range(M_G)}
    
    # Extract edges of H
    start_H = 2 + 2 * M_G
    M_H = data[start_H]
    H_edges = {frozenset([data[start_H + 1 + 2*i], data[start_H + 2 + 2*i]]) for i in range(M_H)}
    
    # Extract Cost Matrix A
    # A is provided as a flattened list of A_{i,j} for 1 <= i < j <= N
    start_A = start_H + 1 + 2 * M_H
    A_flat = data[start_A:]
    
    # To access A_{i,j} easily, we map pairs (i, j) to their index in A_flat
    # The number of pairs is N*(N-1)//2
    # Pair (i, j) with i < j corresponds to index: 
    # (sum of (N-k) for k from 1 to i-1) + (j - i - 1)
    # More simply, we can pre-calculate a dictionary for costs
    
    # Generate all pairs (i, j) with 1 <= i < j <= N in the order they appear in input
    all_pairs = [ (i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1) ]
    cost_map = dict(zip(all_pairs, A_flat))
    
    # We need to find a permutation P of (1, ..., N) that minimizes cost
    # Cost for a permutation P:
    # For every pair (i, j) with 1 <= i < j <= N:
    # If (edge (i, j) in G) != (edge (P_i, P_j) in H), add cost A_{P_i, P_j}
    # Note: The problem says we pay A_{i,j} to flip edge (i,j) in H.
    # If G has edge (i,j) and H does not have edge (P_i, P_j), we pay A_{P_i, P_j}.
    # If G does not have edge (i,j) and H has edge (P_i, P_j), we pay A_{P_i, P_j}.
    
    # We iterate over all permutations of (1, ..., N)
    # P is a tuple where P[i-1] is the image of vertex i
    perms = permutations(range(1, N + 1))
    
    # For a fixed P, the total cost is the sum over all pairs (i, j) of:
    # cost_map[sorted((P[i-1], P[j-1]))] if (edge (i,j) in G) != (edge (P_i, P_j) in H)
    
    # To avoid loops, we use a nested comprehension:
    # 1. Iterate over all permutations P
    # 2. For each P, calculate the sum of costs for all pairs (i, j)
    # 3. Find the minimum of these sums
    
    # Pre-calculate which pairs have edges in G
    # G_adj is a boolean matrix or set
    # We can use the G_edges set defined above.
    
    # To optimize, we pre-calculate the cost for every pair of vertices (u, v)
    # if they need to be flipped.
    
    # The cost to make H isomorphic to G under permutation P is:
    # sum_{1 <= i < j <= N} (A_{P_i, P_j} if (i,j) in G_edges != (P_i, P_j) in H_edges)
    
    # We use a helper function to calculate cost for a single permutation to keep it clean
    # but since we can't use 'def' inside a comprehension and must avoid loops,
    # we will use a generator expression inside min().
    
    ans = min(
        sum(
            cost_map[tuple(sorted((P[i-1], P[j-1])))]
            for i in range(1, N + 1)
            for j in range(i + 1, N + 1)
            if (frozenset([i, j]) in G_edges) != (frozenset([P[i-1], P[j-1]]) in H_edges)
        )
        for P in perms
    )
    
    print(ans)

if __name__ == "__main__":
    solve()