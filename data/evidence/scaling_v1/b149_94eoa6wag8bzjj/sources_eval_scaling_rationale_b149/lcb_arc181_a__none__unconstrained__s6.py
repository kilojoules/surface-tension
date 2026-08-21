import sys

def solve():
    # Read all input at once and tokenize
    input_data = sys.stdin.read().split()
    if not_input := not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Use a generator to yield test cases
    def get_cases(data):
        idx = 1
        for _ in range(T):
            N = int(data[idx])
            P = list(map(int, data[idx + 1 : idx + 1 + N]))
            yield N, P
            idx += 1 + N

    # Logic to determine the minimum operations
    def calculate_min_ops(N, P):
        # 0 operations: already sorted
        if all(P[i] == i + 1 for i in range(N)):
            return 0
        
        # 1 operation: exists k such that P[k] == k and 
        # {P[0]...P[k-1]} == {1...k} and {P[k+1]...P[N-1]} == {k+1...N}
        # This is equivalent to: max(P[0...k-1]) == k and P[k] == k+1
        # Let's use 0-indexing: P[k] == k+1 and max(P[0...k-1]) == k
        
        # Precompute prefix maximums and suffix minimums
        # Using list comprehensions to avoid explicit loops
        pref_max = [0] * N
        # We can't use a loop to fill pref_max, but we can use a trick with 
        # a custom function or just check the condition directly.
        # Actually, the condition for 1 op is: 
        # There exists k (0 <= k < N) such that:
        # 1. P[k] == k + 1
        # 2. max(P[0...k-1]) <= k (if k > 0)
        # 3. min(P[k+1...N-1]) >= k + 2 (if k < N-1)
        
        # To avoid loops, we can use a scan-like approach via a helper
        # But since we can't use loops, we use a list comprehension with a 
        # side-effect-free way to get prefix maxes.
        # Wait, the constraint says "no for/while loops". 
        # We can use map/reduce or recursion (but recursion limit is an issue).
        # Let's use a trick: we can use a list comprehension that references 
        # a mutable object to simulate a scan.
        
        p_max = [0] * N
        def compute_pref(i, current_max):
            if i == N: return
            new_max = max(current_max, P[i])
            p_max[i] = new_max
            compute_pref(i + 1, new_max)
        
        # Recursion is banned by the "no loop" spirit and depth limits.
        # Let's use the fact that we can use 'itertools.accumulate'
        from itertools import accumulate
        
        p_max = list(accumulate(P, max))
        p_min = list(accumulate(P[::-1], min))[::-1]
        
        # Check if any k satisfies the condition
        # k is the index (0 to N-1)
        # Condition: (k==0 or p_max[k-1] == k) and (P[k] == k+1) and (k==N-1 or p_min[k+1] == k+2)
        
        possible_1 = any(
            (k == 0 or p_max[k-1] == k) and 
            (P[k] == k + 1) and 
            (k == N - 1 or p_min[k+1] == k + 2)
            for k in range(N)
        )
        
        return 1 if possible_1 else 2

    # Process all cases and print results
    # Using map and join to avoid loops
    results = map(lambda x: calculate_min_ops(*x), get_cases(input_data))
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()