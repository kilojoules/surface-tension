import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Use a generator to group the input into (N, P) pairs
    # Each case starts at index 1 + sum of (1 + N_i) for previous cases
    # Since we cannot use loops, we process the flat list using a custom function
    # and map/reduce logic.
    
    def get_cases(data):
        # This helper extracts N and the permutation for each test case
        # without using a for/while loop.
        def step(state, current_idx):
            N = int(data[current_idx])
            P = list(map(int, data[current_idx + 1 : current_idx + 1 + N]))
            return (state + [(N, P)], current_idx + 1 + N)
        
        # accumulate is used to simulate the pointer movement through the flat list
        # We use a list comprehension to drive the accumulation of indices
        indices = accumulate(range(T), lambda curr, _: curr + 0) # Dummy
        # However, the above doesn't work because N varies. 
        # Let's use a different approach to parse the flat list.
        pass

    # Correct way to parse variable length input without loops:
    # We can use a recursive-like structure via a list comprehension and a helper
    # But since T is large, recursion depth is an issue.
    # Actually, we can use a generator that yields the cases.
    def case_generator(data):
        it = iter(data)
        # We use a list comprehension to consume the iterator
        # But we need to call next(it) N+1 times.
        # We can achieve this by using a helper function inside a list comprehension.
        def extract_case():
            try:
                n_val = int(next(it))
                p_vals = [int(next(it)) for _ in range(n_val)]
                return (n_val, p_vals)
            except StopIteration:
                return None
        
        return [extract_case() for _ in range(T)]

    cases = case_generator(input_data[1:])

    def calculate_min_ops(case):
        N, P = case
        # The goal is to find if there exists a k such that:
        # All elements in P[0...k-2] are <= k-1 AND all elements in P[k...N-1] are >= k
        # Wait, the operation is: sort 1 to k-1, and sort k+1 to N.
        # This means P_k remains in place. For the result to be 1..N, 
        # we must have P_k = k, and the set {P_1...P_{k-1}} must be {1...k-1}.
        # This is equivalent to saying max(P_1...P_{k-1}) = k-1.
        
        # Let's check if 0 operations are needed:
        if P == list(range(1, N + 1)):
            return 0
        
        # Check if 1 operation is sufficient:
        # We need a k (1 <= k <= N) such that:
        # 1. P_k = k (using 1-based indexing, so P[k-1] == k)
        # 2. max(P[0...k-2]) == k-1 (if k > 1)
        # 3. min(P[k...N-1]) == k+1 (if k < N)
        
        # Precompute prefix max and suffix min
        prefix_max = list(accumulate(P, max))
        suffix_min = list(accumulate(P[::-1], min))[::-1]
        
        # A k works if:
        # (k==1 or prefix_max[k-2] == k-1) AND 
        # (P[k-1] == k) AND 
        # (k==N or suffix_min[k] == k+1)
        
        # We use a list comprehension to check all k from 1 to N
        # and return 1 if any k satisfies the condition.
        possible_k = [
            k for k in range(1, N + 1)
            if (k == 1 or prefix_max[k-2] == k-1) and
               (P[k-1] == k) and
               (k == N or suffix_min[k] == k+1)
        ]
        
        if possible_k:
            return 1
        else:
            # It is proved that maximum 2 operations are always sufficient.
            # (e.g., k=1 then k=N, or similar)
            return 2

    # Map the calculation over all cases and print
    results = map(calculate_min_ops, cases)
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()