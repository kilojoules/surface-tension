import sys

def solve():
    # Read all input at once and split into a flat list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use an iterator to consume tokens one by one
    it = iter(input_data)
    T_str = next(it, None)
    if T_str is None:
        return
    T = int(T_str)
    
    # Process each test case using a generator expression
    # The logic:
    # 0 operations: Already sorted.
    # 1 operation: There exists k such that sorting [1, k-1] and [k+1, N] fixes everything.
    #   This is possible if there is some k where P[k] is the only element 
    #   not in its sorted position relative to the others, or more simply,
    #   if we can pick k such that all elements {1...k-1} are in positions 1...k-1
    #   (after sorting) and {k+1...N} are in positions k+1...N.
    #   Actually, 1 operation is possible if there is a k such that:
    #   {P_1, ..., P_{k-1}, P_{k+1}, ..., P_N} = {1, ..., k-1, k+1, ..., N}
    #   Which is always true for any k. The real condition is:
    #   Can we pick k such that after sorting the two partitions, P_i = i?
    #   This happens if P_k = k and the remaining elements are just a permutation
    #   of the remaining values. But the operation sorts them!
    #   So 1 operation is possible if there exists k such that:
    #   The set of values {P_1, ..., P_{k-1}} union {P_{k+1}, ..., P_N} 
    #   is exactly {1, ..., k-1, k+1, ..., N}.
    #   This is true if and only if P_k = k.
    #   Wait, if P_k = k, then sorting the left and right sides will 
    #   automatically put every element i in position i.
    #   However, we need to check if the permutation is already sorted (0).
    #   If not sorted, can we do it in 1? Yes, if there is at least one k where P_k = k.
    #   If for all k, P_k != k, we need 2 operations.
    #   (It is proven that 2 is the maximum needed for N >= 3).
    
    def process_case():
        N = int(next(it))
        P = [int(next(it)) for _ in range(N)]
        
        # Check if already sorted
        # Using all() in a generator for efficiency
        is_sorted = all(P[i] == i + 1 for i in range(N))
        if is_sorted:
            return 0
        
        # Check if there is any k such that P[k] == k+1
        # If such k exists, one operation at k fixes the permutation.
        has_fixed_point = any(P[i] == i + 1 for i in range(N))
        return 1 if has_fixed_point else 2

    # Use map and join to output all results without explicit for-loops
    results = map(lambda _: process_case(), range(T))
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()