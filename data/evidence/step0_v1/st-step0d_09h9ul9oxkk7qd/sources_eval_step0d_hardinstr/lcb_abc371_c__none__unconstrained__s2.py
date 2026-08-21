import itertools
import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    it = iter(input_data)
    N = int(next(it))
    
    # Graph G adjacency matrix
    MG = int(next(it))
    G_edges = [tuple(map(int, (next(it), next(it)))) for _ in range(MG)]
    G_adj = [[0] * N for _ in range(N)]
    # Using a list comprehension to populate G_adj is tricky without loops, 
    # but we can define it based on whether (i, j) is in G_edges.
    # To avoid loops, we pre-process G_edges into a set.
    G_set = set(tuple(sorted(e)) for e in G_edges)
    
    # Graph H adjacency matrix
    MH = int(next(it))
    H_edges = [tuple(map(int, (next(it), next(it)))) for _ in range(MH)]
    H_set = set(tuple(sorted(e)) for e in H_edges)
    
    # Cost matrix A
    # A is given as A_{1,2}, A_{1,3}... A_{N-1,N}
    # We store it in a dictionary for easy access: (i, j) -> cost
    all_costs = list(map(int, it))
    
    # Map the flat cost list to pairs (i, j)
    # There are N*(N-1)//2 pairs.
    pairs = list(itertools.combinations(range(1, N + 1), 2))
    cost_map = dict(zip(pairs, all_costs))

    # We want to find a permutation P of (1...N) such that 
    # the cost to make H match G under P is minimized.
    # The cost for a pair (i, j) is A_{P_i, P_j} if 
    # (edge in G between i,j) != (edge in H between P_i, P_j)
    
    # Pre-calculate G's adjacency for all pairs
    # G_status(i, j) is 1 if edge exists, 0 otherwise
    # We use 0-indexing for the permutation logic
    
    def get_cost(P):
        # P is a permutation of (0, ..., N-1)
        # We check all pairs (i, j) where 0 <= i < j < N
        # The corresponding vertices in H are P[i]+1 and P[j]+1
        return sum(
            cost_map[tuple(sorted((P[i] + 1, P[j] + 1)))]
            for i, j in itertools.combinations(range(N), 2)
            if ((i + 1, j + 1) in G_set) != (tuple(sorted((P[i] + 1, P[j] + 1))) in H_set)
        )

    # Try all permutations of vertices
    ans = min(get_cost(P) for P in itertools.permutations(range(N)))
    print(ans)

if __name__ == "__main__":
    solve()