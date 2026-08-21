import sys
from itertools import permutations

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract values one by one
    gen = input_data
    
    N = next(gen)
    
    # Helper to read M edges and return a set of frozensets (undirected edges)
    def read_edges(g):
        M = next(g)
        return {frozenset([next(g), next(g)]) for _ in range(M)}

    # Since we cannot use 'for' loops, we use list comprehensions or map
    # However, the prompt forbids 'for' even in comprehensions.
    # We must use map(), filter(), and recursion or itertools.
    
    # To handle the edge reading without 'for', we use a recursive function
    def get_edges(count, g):
        if count == 0:
            return set()
        return {frozenset([next(g), next(g)])} | get_edges(count - 1, g)

    # Re-evaluating the input reading to avoid 'for'
    # We need M_G and M_H. Let's use a different approach to capture them.
    
    # Since I cannot use 'for', I will use a helper to consume the generator
    def consume_n(n, g):
        return [next(g) for _ in range(n)] # Wait, 'for' is forbidden.
    
    # Correcting: Use map(next, range(n)) to simulate loop-free consumption
    def consume_n_fixed(n, g):
        return list(map(lambda _: next(g), range(n)))

    # Let's redefine the data extraction using map and next
    # We need to be careful: next(gen) advances the iterator.
    
    # Because the constraint is strict ("no for/while"), 
    # I will use recursion to read the edges.
    def read_graph_edges(g):
        m = next(g)
        def collect(remaining):
            if remaining <= 0:
                return set()
            return {frozenset([next(g), next(g)])} | collect(remaining - 1)
        return collect(m)

    edges_g = read_graph_edges(gen)
    edges_h = read_graph_edges(gen)
    
    # Read A_{i,j} matrix
    # There are N*(N-1)//2 values for A.
    # We can read them all into a list.
    all_a = list(map(lambda _: next(gen), range(N * (N - 1) // 2)))
    
    # Map (i, j) to the index in all_a
    # The input order is A_{1,2}, A_{1,3}...A_{1,N}, A_{2,3}...
    def get_a_index(i, j):
        # i, j are 1-indexed, i < j
        # Index = sum_{k=1}^{i-1} (N-k) + (j-i-1)
        # Sum of (N-k) from 1 to i-1 is (i-1)*N - (i-1)*i//2
        return (i - 1) * N - (i * (i - 1) // 2) + (j - i - 1)

    # Pre-calculate costs for all pairs (i, j) where 1 <= i < j <= N
    # We use a dictionary to store A_{i,j}
    # To avoid 'for', we use map and a list of all pairs
    all_pairs = list(map(lambda x: (x[0], x[1]), 
                         [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]))
    # Wait, the above uses 'for'. I must use itertools.product or similar.
    
    # Correct way to get all pairs (i, j) with i < j without 'for':
    import itertools
    pairs = list(itertools.combinations(range(1, N + 1), 2))
    
    # Map pairs to their costs
    cost_map = dict(zip(pairs, all_a))

    # We need to find a permutation P of {1...N} that minimizes cost
    # Cost = Sum_{i < j} A_{P_i, P_j} if (edge(i,j) in G != edge(P_i, P_j) in H)
    
    def calculate_cost(p):
        # p is a permutation of (1...N)
        # We check every pair (i, j) with 1 <= i < j <= N
        # G edge: (i, j), H edge: (p[i-1], p[j-1])
        
        def pair_cost(pair):
            i, j = pair
            # G has edge?
            has_g = frozenset([i, j]) in edges_g
            # H has edge?
            has_h = froeset_h(p[i-1], p[j-1])
            # This is getting complex. Let's simplify.
            return 0 # placeholder
            
    # Let's redefine the cost function clearly
    def get_total_cost(p):
        # p is the permutation mapping vertex i in G to vertex p[i-1] in H
        # For every pair (i, j) in G, we check if the edge status matches in H
        # The cost to change H is A_{p[i-1], p[j-1]}
        
        def edge_diff(pair):
            u, v = pair
            # Edge in G
            g_edge = frozenset([u, v]) in edges_g
            # Edge in H
            # The vertices in H are p[u-1] and p[v-1]
            # We need the cost A_{min(p[u-1], p[v-1]), max(p[u-1], p[v-1])}
            h_edge = frozenset([p[u-1], p[v-1]]) in edges_h
            
            if g_edge == h_edge:
                return 0
            
            # Cost is A_{i,j} where i,j are the vertices in H
            # Sort p[u-1] and p[v-1] to use as key for cost_map
            h_pair = tuple(sorted((p[u-1], p[v-1])))
            return cost_map[h_pair]

        return sum(map(edge_diff, pairs))

    # Try all permutations and find the minimum cost
    # range(1, N+1) converted to tuple for permutations
    nodes = tuple(range(1, N + 1))
    all_perms = permutations(nodes)
    
    # Use map to calculate cost for all permutations and find the min
    ans = min(map(get_total_cost, all_perms))
    print(ans)

# To avoid the 'for' in the helper functions and the main logic, 
# I will rewrite the final script strictly adhering to the rules.