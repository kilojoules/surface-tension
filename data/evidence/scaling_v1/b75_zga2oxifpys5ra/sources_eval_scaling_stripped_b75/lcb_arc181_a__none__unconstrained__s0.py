import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Use a generator to process test cases without loops
    # We need to group the flat input list into (N, P) pairs
    # Since N varies, we can't use a simple chunk size.
    # However, we can use a custom function with accumulate to track the current index.
    
    def get_cases(data):
        # state: (current_index, list_of_cases)
        def step(state, _):
            idx, cases = state
            if idx >= len(data):
                return state
            n = int(data[idx])
            p = list(map(int, data[idx+1 : idx+1+n]))
            return (idx + 1 + n, cases + [(n, p)])
        
        # We run accumulate T times to extract T cases
        final_state = list(accumulate([None] * T, step, initial=(1, [])))[-1]
        return final_state[1]

    cases = get_cases(input_data)

    def calculate_min_ops(case):
        n, p = case
        # The goal is to find if there exists a k such that:
        # 1. All elements in P[0...k-2] are <= k-1 (after sorting, they become 1...k-1)
        # 2. All elements in P[k...n-1] are >= k+1 (after sorting, they become k+1...n)
        # This is equivalent to:
        # max(P[0...k-2]) <= k-1 AND min(P[k...n-1]) >= k+1
        # Note: k is 1-indexed in the problem, so P[k-1] is the pivot.
        
        # Precompute prefix maximums and suffix minimums
        # prefix_max[i] = max(P[0...i])
        prefix_max = list(accumulate(p, max))
        # suffix_min[i] = min(P[i...n-1])
        suffix_min = list(accumulate(p[::-1], min))[::-1]
        
        # Check if already sorted
        if p == sorted(p):
            return 0
        
        # Check if 1 operation is enough:
        # For a given k (1 <= k <= n):
        # If k=1: check suffix_min[1] >= 2
        # If k=n: check prefix_max[n-2] <= n-1
        # If 1 < k < n: check prefix_max[k-2] <= k-1 and suffix_min[k] >= k+1
        
        # We use a list comprehension to check all k and 'any()' to see if one works.
        # Using a helper to avoid index errors for k=1 and k=n.
        def check(k):
            # k is 1-indexed
            cond1 = (k == 1) or (prefix_max[k-2] <= k-1)
            cond2 = (k == n) or (suffix_min[k] >= k+1)
            return cond1 and cond2

        # If any k works, answer is 1, otherwise 2.
        # (It is proven that max 2 operations are always sufficient for N >= 3)
        return 1 if any(map(check, range(1, n + 1))) else 2

    # Map the solve function over all cases and print
    results = map(calculate_min_ops, cases)
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()