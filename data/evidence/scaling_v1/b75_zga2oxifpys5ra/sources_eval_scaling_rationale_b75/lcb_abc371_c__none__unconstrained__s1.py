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
    
    # Read Graph G edges
    Mg = int(next(it))
    g_edges = [tuple(sorted((int(next(it)), int(next(it))))) for _ in range(Mg)]
    
    # Read Graph H edges
    Mh = int(next(it))
    h_edges = [tuple(sorted((int(next(it)), int(next(it))))) for _ in range(Mh)]
    
    # Read Cost Matrix A
    # A is given as a flattened upper triangle
    # We store it in a dictionary for O(1) access: (i, j) -> cost
    # The input order is A_{1,2}, A_{1,3}... A_{1,N}, A_{2,3}...
    flat_costs = [int(next(it)) for _ in range(N * (N - 1) // 2)]
    
    # To map the flat list to (i, j) pairs:
    # We can use a helper to generate the indices
    cost_map = {
        (i, j): flat_costs[sum(N - k for k in range(1, i)) + (j - i - 1)]
        for i in range(1, N + 1)
        for j in range(i + 1, N + 1)
    }

    # Represent G and H as adjacency matrices (sets of edges for fast lookup)
    g_set = set(g_edges)
    h_set = set(h_edges)

    # We need to find a permutation P of (1...N) such that 
    # we minimize the cost of changing H to match G under mapping P.
    # The cost is the sum of A_{i,j} for all pairs (i,j) where 
    # the edge status of (P_i, P_j) in H differs from (i, j) in G.
    
    # Generate all permutations of vertices 1...N
    all_perms = permutations(range(1, N + 1))
    
    # For a given permutation P, the cost is:
    # Sum_{1 <= i < j <= N} A_{P_i, P_j} if (edge(i,j) in G != edge(P_i, P_j) in H)
    # Note: The problem says we pay A_{i,j} to flip edge (i,j) in H.
    # If we map vertex i of G to vertex P_i of H, then edge (i,j) in G 
    # corresponds to edge (P_i, P_j) in H.
    
    # We use a helper function to get cost for a pair regardless of order
    get_cost = lambda u, v: cost_map.get(tuple(sorted((u, v))), 0)

    # We use a list comprehension to calculate total cost for every permutation
    # and then find the minimum.
    ans = min(
        sum(
            get_cost(p[i-1], p[j-1])
            for i in range(1, N + 1)
            for j in range(i + 1, N + 1)
            if ((i, j) in g_set) != (tuple(sorted((p[i-1], p[j-1]))) in h_set)
        )
        for p in all_perms
    )
    
    print(ans)

if __name__ == "__main__":
    solve()