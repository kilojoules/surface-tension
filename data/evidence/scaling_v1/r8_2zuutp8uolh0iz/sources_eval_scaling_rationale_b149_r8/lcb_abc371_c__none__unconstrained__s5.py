import itertools
import sys

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use a pointer-like approach via a list to simulate reading
    # Since we can't use while loops, we map the input to variables
    # based on the known structure.
    
    N = int(input_data[0])
    
    # M_G is at index 1
    MG = int(input_data[1])
    # Edges of G are from index 2 to 2 + 2*MG - 1
    g_edges_raw = input_data[2 : 2 + 2*MG]
    
    # M_H is at index 2 + 2*MG
    MH_idx = 2 + 2*MG
    MH = int(input_data[MH_idx])
    # Edges of H are from MH_idx + 1 to MH_idx + 1 + 2*MH - 1
    h_edges_raw = input_data[MH_idx + 1 : MH_idx + 1 + 2*MH]
    
    # A values start after the H edges
    A_raw = input_data[MH_idx + 1 + 2*MH :]
    
    # Represent G and H as adjacency matrices (0 or 1)
    # Using list comprehensions to build the matrices
    G = [[0] * N for _ in range(N)]
    # Since we can't use for loops to fill G, we recreate it 
    # by checking if (i, j) is in the edge list.
    g_edge_set = set(
        tuple(sorted((int(g_edges_raw[i]), int(g_edges_raw[i+1]))))
        for i in range(0, len(g_edges_raw), 2)
    )
    
    h_edge_set = set(
        tuple(sorted((int(h_edges_raw[i]), int(h_edges_raw[i+1]))))
        for i in range(0, len(h_edges_raw), 2)
    )
    
    # Parse A into a 2D structure A[i][j]
    # The input for A is given as A_{1,2}, A_{1,3}... A_{N-1,N}
    # We can map these to a dictionary using a list comprehension
    # to avoid for loops.
    
    # Generate all pairs (i, j) with 1 <= i < j <= N
    all_pairs = [
        (i, j) 
        for i in range(1, N + 1) 
        for j in range(i + 1, N + 1)
    ]
    
    # Map each pair to its cost
    cost_map = dict(zip(all_pairs, map(int, A_raw)))
    
    # We want to find a permutation P of (1...N) such that 
    # we minimize the cost to make H isomorphic to G.
    # The cost for a permutation P is the sum of A_{P_i, P_j} 
    # for all pairs (i, j) where (i, j) is an edge in G 
    # but (P_i, P_j) is NOT an edge in H, OR vice versa.
    
    # Pre-calculate the cost for every pair (u, v) if their 
    # edge status needs to be toggled.
    # Let's define a function to get cost of pair (u, v)
    # Note: u, v are 1-indexed.
    
    def get_cost(u, v, is_edge_in_g):
        # Sort u, v to match cost_map keys
        pair = tuple(sorted((u, v)))
        # Check if (u, v) is an edge in H
        is_edge_in_h = pair in h_edge_set
        # If status differs, we pay A_{u,v}, otherwise 0
        return cost_map[pair] if is_edge_in_g != is_edge_in_h else 0

    # Iterate through all permutations of (1...N)
    # For each permutation P, calculate total cost.
    # P[i-1] is the vertex in H that vertex i in G is mapped to.
    
    all_perms = itertools.permutations(range(1, N + 1))
    
    # For a fixed permutation P, the total cost is:
    # Sum_{1 <= i < j <= N} get_cost(P[i-1], P[j-1], (i, j) in g_edge_set)
    
    # To avoid nested loops, we use a generator expression inside sum()
    # and a list comprehension to evaluate all permutations.
    
    ans = min(
        sum(
            get_cost(p[i], p[j], (i + 1, j + 1) in g_edge_set)
            for i in range(N)
            for j in range(i + 1, N)
        )
        for p in all_perms
    )
    
    print(ans)

if __name__ == "__main__":
    solve()