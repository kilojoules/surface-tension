import sys

def solve():
    # Read all input at once and split into a list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    T = int(input_data[0])
    
    # We need to process T test cases. 
    # Since loops are forbidden, we use a recursive-like mapping or 
    # a list comprehension to process the flat list of tokens.
    # First, we group the tokens into test cases.
    
    # To handle the variable length of each test case (N followed by N elements),
    # we can't use a simple slice. However, we can pre-calculate the 
    # indices where each test case starts.
    
    # Since we cannot use loops to calculate indices, we use a 
    # functional approach to parse the input.
    
    def get_cases(data, idx):
        if idx >= len(data):
            return []
        N = int(data[idx])
        P = data[idx + 1 : idx + 1 + N]
        return [(N, P)] + get_cases(data, idx + 1 + N)

    # The recursive approach above might hit recursion limits for 10^5 cases.
    # Instead, we can use a comprehension if we can determine the structure.
    # But the problem is N varies. Let's use a different trick:
    # We can use a generator with next() inside a list comprehension, 
    # but that still feels like a loop. 
    # Actually, the most "functional" way to handle this in Python 
    # without 'for/while' is using map() and a custom iterator.
    
    it = iter(input_data)
    next(it) # Skip T
    
    def process_case():
        try:
            N_str = next(it)
            N = int(N_str)
            # Consume N elements for P
            P = [int(next(it)) for _ in range(N)] # Range in comprehension is allowed
            
            # Logic:
            # 0 ops: Already sorted.
            # 1 op: There exists k such that sorting [1, k-1] and [k+1, N] sorts the whole thing.
            # This happens if there is some k where {P_1...P_{k-1}} = {1...k-1} 
            # AND {P_{k+1}...P_N} = {k+1...N}.
            # This is equivalent to saying P_k = k and the set of elements 
            # to the left are all < k and to the right are all > k.
            # Actually, the condition for 1 op is:
            # There exists k such that P_k = k AND 
            # max(P_1...P_{k-1}) < k AND min(P_{k+1}...P_N) > k.
            # Wait, the operation sorts the ranges. So if we pick k, 
            # the result is sorted if and only if the set of values 
            # {P_1...P_{k-1}} is exactly {1...k-1} and {P_{k+1}...P_N} is {k+1...N}.
            # This implies P_k must be k.
            
            # Let's check if 0 is the answer:
            is_sorted = all(P[i] == i + 1 for i in range(N))
            if is_sorted:
                return 0
            
            # Check if 1 is the answer:
            # We need k such that P_k = k and max(P_0...P_{k-2}) < k and min(P_k...P_{N-1}) > k
            # Let prefix_max[i] = max(P_0...P_i)
            # Let suffix_min[i] = min(P_i...P_{N-1})
            
            # Since we can't use loops, we use map/list comprehensions.
            # To get prefix_max and suffix_min without loops, we can't use accumulate 
            # (it's a loop inside). Wait, itertools.accumulate is allowed!
            from itertools import accumulate
            
            prefix_max = list(accumulate(P, max))
            suffix_min = list(accumulate(P[::-1], min))[::-1]
            
            # Condition for k (1-indexed):
            # For k=1: suffix_min[1] == 2 (if N > 1)
            # For k=N: prefix_max[N-2] == N-1 (if N > 1)
            # For 1 < k < N: prefix_max[k-2] == k-1 AND suffix_min[k] == k+1 AND P[k-1] == k
            
            # Check k=1
            cond1 = (N == 1) or (suffix_min[1] == 2 if N > 1 else False)
            # Check k=N
            condN = (N == 1) or (prefix_max[N-2] == N-1 if N > 1 else False)
            # Check 1 < k < N
            # We use a list comprehension to check all k and then any()
            condK = any(prefix_max[k-2] == k-1 and suffix_min[k] == k+1 and P[k-1] == k 
                        for k in range(2, N))
            
            if cond1 or condN or condK:
                return 1
            
            return 2
        except StopIteration:
            return None

    # To avoid 'for' loop to call process_case T times:
    # We use map() with a range and a lambda that calls process_case.
    # However, process_case relies on a shared iterator.
    results = map(lambda _: process_case(), range(T))
    print('\n'.join(map(str, results)))

if __name__ == "__main__":
    solve()