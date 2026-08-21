import itertools
import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use a pointer-like approach via an iterator to consume input without indices
    it = iter(input_data)
    
    N = int(next(it))
    
    # Graph G edges
    MG = int(next(it))
    G_edges = [tuple(sorted((int(next(it)), int(next(it))))) for _ in range(MG)]
    
    # Graph H edges
    MH = int(next(it))
    H_edges = [tuple(sorted((int(next(it)), int(next(it))))) for _ in range(MH)]
    
    # Cost matrix A
    # A[i][j] will store cost for edge (i+1, j+1)
    # We read the costs and organize them into a dictionary for O(1) access
    # The input format for A is A_{1,2}, A_{1,3}... A_{1,N}, A_{2,3}...
    # We can use a list comprehension to flatten the remaining input and 
    # map them to pairs (i, j)
    all_costs = [int(x) for x in it]
    
    # Generate all possible pairs (i, j) with 1 <= i < j <= N
    pairs = list(itertools.combinations(range(1, N + 1), 2))
    cost_map = dict(zip(pairs, all_costs))
    
    # Adjacency matrices for G and H for O(1) lookup
    # Using sets of tuples for edges
    set_G = set(G_edges)
    set_H = set(H_edges)
    
    # We need to find a permutation P of (1...N) such that 
    # the cost to make H isomorphic to G is minimized.
    # The cost for a permutation P is the sum of A_{P_i, P_j} 
    # for all pairs (i, j) where (edge in G) != (edge in H after mapping)
    # Wait, the problem says we modify H. 
    # If we map vertex i of G to vertex P_i of H:
    # An edge (i, j) in G must exist in H between P_i and P_j.
    # If it doesn't, we pay A_{P_i, P_j} to add it.
    # If (i, j) is NOT an edge in G, but (P_i, P_j) is an edge in H,
    # we pay A_{P_i, P_j} to remove it.
    
    # Pre-calculate all pairs (i, j) for G
    g_pairs = list(itertools.combinations(range(1, N + 1), 2))
    
    # The permutation P maps G-vertex -> H-vertex
    # We iterate through all permutations of (1...N)
    perms = itertools.permutations(range(1, N + 1))
    
    # For a fixed permutation P, the total cost is:
    # sum(cost_map[sorted((P[i-1], P[j-1]))] 
    #     for (i, j) in g_pairs 
    #     if ( (i, j) in set_G ) != ( (sorted((P[i-1], P[j-1]))) in set_H )
    #    )
    
    # To avoid loops, we use a generator expression inside min()
    # We use a helper function to get the sorted pair to keep the comprehension clean
    get_pair = lambda x, y: (x, y) if x < y else (y, x)
    
    ans = min(
        sum(
            cost_map[get_pair(p[i-1], p[j-1])]
            for i, j in g_pairs
            if ((i, j) in set_G) != (get_pair(p[i-1], p[j-1]) in set_H)
        )
        for p in perms
    )
    
    print(ans)

if __name__ == "__main__":
    solve()