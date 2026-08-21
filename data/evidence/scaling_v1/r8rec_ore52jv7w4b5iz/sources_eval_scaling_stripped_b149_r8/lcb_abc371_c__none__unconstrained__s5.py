import itertools
import sys

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use a generator/iterator to consume input sequentially
    it = iter(map(int, input_data))
    
    N = next(it)
    
    # Read Graph G
    M_G = next(it)
    # Create a set of edges for G. Each edge is a sorted tuple (u, v)
    # We use a list comprehension to consume 2*M_G elements from the iterator
    g_edges_list = [next(it) for _ in range(2 * M_G)]
    G = {tuple(sorted((g_edges_list[i], g_edges_list[i+1]))) 
         for i in range(0, 2 * M_G, 2)}
    
    # Read Graph H
    M_H = next(it)
    h_edges_list = [next(it) for _ in range(2 * M_H)]
    H = {tuple(sorted((h_edges_list[i], h_edges_list[i+1]))) 
         for i in range(0, 2 * M_H, 2)}
    
    # Read Cost Matrix A
    # A[i][j] will store the cost to flip edge (i+1, j+1)
    # The input provides A_{1,2}, A_{1,3}... A_{1,N}, then A_{2,3}...
    # We read the remaining items into a list
    flat_A = list(it)
    
    # To map (i, j) to the index in flat_A:
    # The number of elements before A_{i,j} (1 <= i < j <= N) is:
    # Sum_{k=1}^{i-1} (N - k)  + (j - i - 1)
    # Sum_{k=1}^{i-1} (N - k) = (i-1)*N - (i-1)*i/2
    
    # We pre-calculate the cost for every possible pair (i, j) 1 <= i < j <= N
    # cost_map[(i, j)] = cost
    # We can use a list comprehension to build the map
    # The total number of pairs is N*(N-1)//2
    pairs = [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]
    cost_map = {pairs[idx]: flat_A[idx] for idx in range(len(pairs))}

    # We need to find a permutation P of (1...N) that minimizes:
    # Sum_{1 <= i < j <= N} cost(P_i, P_j) if (edge(i,j) in G != edge(P_i, P_j) in H)
    
    # Generate all permutations of (1...N)
    perms = itertools.permutations(range(1, N + 1))
    
    # For a fixed permutation P, the cost is the sum over all pairs (i, j)
    # where the adjacency in G(i, j) differs from adjacency in H(P_i, P_j)
    # Note: i and j are indices 1...N
    
    # To avoid loops, we use a generator expression inside min()
    # We iterate over all permutations P
    # For each P, we calculate the sum of costs for all pairs (i, j)
    
    # We define the cost for a specific permutation P
    # P is a tuple where P[i-1] is the image of vertex i
    
    ans = min(
        sum(
            cost_map[tuple(sorted((P[i-1], P[j-1])))]
            for i in range(1, N + 1)
            for j in range(i + 1, N + 1)
            if ((i, j) in G) != (tuple(sorted((P[i-1], P[j-1]))) in H)
        )
        for P in perms
    )
    
    print(ans)

if __name__ == "__main__":
    solve()