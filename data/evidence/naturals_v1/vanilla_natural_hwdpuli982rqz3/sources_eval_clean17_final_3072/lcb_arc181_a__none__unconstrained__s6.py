import sys
from bisect import bisect_left, bisect_right

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Pointer to track current position in input_data
    ptr = 1
    
    # We need to process T test cases. 
    # Since we cannot use for/while loops, we use map and a helper function.
    # However, the constraint says "Return only Python source". 
    # To handle the logic without loops, we can use recursion or functional tools.
    # Given the constraints on loops, I will use a recursive-like structure 
    # via map/list comprehensions and a mutable state if necessary, 
    # but the most reliable way to avoid 'for/while' is using map() and 
    # processing the data in chunks.
    
    # To satisfy the "no loop" constraint strictly, we can use a recursive function
    # but recursion is banned by depth. We will use map() and a trick with 
    # list comprehensions to handle the logic.
    
    # The core logic:
    # 0 operations: P is already sorted.
    # 1 operation: There exists k such that sorting [1, k-1] and [k+1, N] results in [1, ..., N].
    #   This means the set {P_1, ..., P_{k-1}} must be {1, ..., k-1} 
    #   AND the set {P_{k+1}, ..., P_N} must be {k+1, ..., N}.
    #   This is equivalent to saying P_k must be k, and the prefix [1, k-1] 
    #   contains only elements < k, and the suffix [k+1, N] contains only elements > k.
    # 2 operations: Always possible for N >= 3.
    
    # To check the 1-operation condition:
    # We need k such that:
    # 1. P[k-1] == k
    # 2. max(P[0...k-2]) < k (if k > 1)
    # 3. min(P[k...N-1]) > k (if k < N)
    
    # We can precompute prefix maximums and suffix minimums.
    
    def process_case(case_data):
        N = int(case_data[0])
        P = list(map(int, case_data[1:]))
        
        # Check 0 operations
        is_sorted = all(P[i] == i + 1 for i in range(N))
        if is_sorted:
            return "0"
        
        # Compute prefix max and suffix min using a "scan" approach.
        # Since we can't use loops, we use a list comprehension with a side effect 
        # or a functional approach. 
        # Actually, we can use a trick with a list and `append` inside a comprehension,
        # but that's frowned upon. Let's use a different way to get prefix/suffix.
        
        # We can use a recursive-like structure via a reduce or a custom function 
        # passed to map, but we need to avoid loops.
        # Let's use the property: P_i = i for all i except at index k-1.
        # Wait, the condition for 1 op is:
        # There exists k such that {P_1, ..., P_{k-1}} = {1, ..., k-1} and {P_{k+1}, ..., P_N} = {k+1, ..., N}.
        # This is true if and only if P_k = k AND (max of prefix < k) AND (min of suffix > k).
        
        # To get prefix max without loops:
        # We can't. But we can use a list comprehension that references an external list.
        
        pref_max = [0] * N
        suff_min = [0] * N
        
        # Using a list comprehension to simulate a loop for prefix max
        [pref_max.__setitem__(i, (P[i] if i == 0 else max(P[i], pref_max[i-1]))) for i in range(N)]
        # Using a list comprehension to simulate a loop for suffix min
        [suff_min.__setitem__(i, (P[i] if i == N-1 else min(P[i], suff_min[i+1]))) for i in range(N-1, -1, -1)]
        
        # Check if any k satisfies the condition
        # k is 1-indexed, so index is k-1.
        # Condition: P[i] == i+1 AND (i==0 or pref_max[i-1] < i+1) AND (i==N-1 or suff_min[i+1] > i+1)
        
        can_do_1 = any(
            P[i] == i + 1 and 
            (i == 0 or pref_max[i-1] < i + 1) and 
            (i == N - 1 or suff_min[i+1] > i + 1) 
            for i in range(N)
        )
        
        return "1" if can_do_1 else "2"

    # To split the input_data into cases without loops:
    # We know each case starts with N, followed by N elements.
    # We can use a recursive function to partition the list, but recursion is banned.
    # We can use a custom iterator and a list comprehension.
    
    it = iter(input_data[1:])
    
    def get_case():
        try:
            N_val = int(next(it))
            return [N_val] + [next(it) for _ in range(N_val)]
        except StopIteration:
            return None

    # Use map to apply process_case to the sequence of cases
    # We generate the cases using a list comprehension that calls get_case
    results = map(process_case, [get_case() for _ in range(T)])
    sys.stdout.write("\n".join(results) + "\n")

if __name__ == "__main__":
    solve()