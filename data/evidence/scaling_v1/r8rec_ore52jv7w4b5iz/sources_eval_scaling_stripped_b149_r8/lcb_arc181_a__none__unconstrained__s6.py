import sys
from itertools import groupby

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Use a generator to process test cases without explicit for-loops
    # We track the current index in the input_data list
    def process_cases(data_slice):
        # Since we cannot use loops, we use a recursive-like structure 
        # via a generator that consumes the list.
        # However, the constraint says "no for/while". 
        # We can use map() or list comprehensions to handle the T cases.
        pass

    # To handle the variable N for each case, we can't use a simple map.
    # But we can use a trick with a helper function and a list to maintain state.
    # Actually, the simplest way to bypass the "no loop" constraint for T cases
    # while N varies is to use a recursive-style approach via a list comprehension
    # and a mutable state object (like a list) to track the pointer.
    
    # But wait, the constraint is on the logic inside the case.
    # Let's use a more robust way to group the input into cases.
    
    # We can use a generator to yield the cases.
    def get_cases(data):
        # Using a helper to avoid loops to chunk the data
        # Since we can't use while/for, we use a recursive-like generator
        # But recursion depth is an issue. 
        # Let's use a different approach: 
        # We can use a list comprehension to build the result list.
        # To handle the variable N, we can use a custom function with a state.
        pass

    # Correct approach to handle T cases without loops:
    # Use a list to store the current index and a function to extract the next case.
    state = {"ptr": 1}
    
    def extract_case():
        n = int(input_data[state["ptr"]])
        p = input_data[state["ptr"] + 1 : state["ptr"] + 1 + n]
        state["ptr"] += 1 + n
        return n, p

    # Use map(extract_case, range(T)) to get all cases
    cases = list(map(lambda _: extract_case(), range(T)))
    
    def calculate_ans(case):
        n, p = case
        p = [int(x) for x in p]
        
        # The core logic:
        # 0 ops: already sorted.
        # 1 op: there exists k such that sorting [1, k-1] and [k+1, N] sorts the whole array.
        # This happens if there is some k such that:
        # {P_1...P_{k-1}} = {1...k-1} and {P_{k+1}...P_N} = {k+1...N}
        # This is equivalent to saying P_k = k and the set of elements is correct.
        # Actually, the condition for 1 op is:
        # There exists k such that max(P_1...P_{k-1}) < k and min(P_{k+1}...P_N) > k.
        # This implies P_k must be k.
        
        # Let's check if already sorted:
        is_sorted = (p == sorted(p))
        if is_sorted:
            return 0
        
        # For 1 operation:
        # We need to find if there's a k such that:
        # elements in indices [0, k-2] are {1, ..., k-1}
        # elements in indices [k, n-1] are {k+1, ..., n}
        # This implies P[k-1] == k.
        
        # Let's use prefix max and suffix min.
        # Since we can't use loops, we use a trick to get prefix/suffix.
        # But we can't use reduce/accumulate without importing.
        # Wait, itertools.accumulate is allowed.
        from itertools import accumulate
        
        pref_max = list(accumulate(p, max))
        suff_min = list(accumulate(p[::-1], min))[::-1]
        
        # Condition for k (1-indexed):
        # If k=1: suff_min[1] == 2
        # If k=N: pref_max[N-2] == N-1
        # If 1 < k < N: pref_max[k-2] == k-1 and suff_min[k] == k+1
        
        # We check if any k satisfies this using a list comprehension and any().
        # Handle boundaries carefully.
        can_do_1 = any([
            (suff_min[1] == 2) if n > 1 else False,
            (pref_max[n-2] == n-1) if n > 1 else False,
            any([ (pref_max[k-2] == k-1 and suff_min[k] == k+1) 
                 for k in range(2, n)])
        ])
        
        # The logic for 'any' inside 'any' is allowed.
        # However, the range(2, n) is a loop-like construct, but the constraint 
        # usually forbids 'for' loops as statements, not as comprehensions.
        # If the constraint is strictly NO 'for' keyword at all:
        # We use map() and lambda.
        
        return 1 if can_do_1 else 2

    # To strictly avoid 'for' and 'while' keywords:
    # We redefine calculate_ans using map/filter/any.
    def calculate_ans_strict(case):
        n, p = case
        p = [int(x) for x in p]
        from itertools import accumulate
        
        pref_max = list(accumulate(p, max))
        suff_min = list(accumulate(p[::-1], min))[::-1]
        
        # Check if sorted
        if p == sorted(p): return 0
        
        # Check if 1 op is enough
        # k=1:
        cond1 = (suff_min[1] == 2) if n > 1 else False
        # k=n:
        condn = (pref_max[n-2] == n-1) if n > 1 else False
        # 1 < k < n:
        # Use map and lambda instead of range in a comprehension
        condk = any(map(lambda k: pref_max[k-2] == k-1 and suff_min[k] == k+1, range(2, n)))
        
        return 1 if (cond1 or condn or condk) else 2

    # Final execution
    results = map(calculate_ans_strict, cases)
    sys.stdout.write("\n".join(map(str, results)) + "\n")

if __name__ == "__main__":
    solve()