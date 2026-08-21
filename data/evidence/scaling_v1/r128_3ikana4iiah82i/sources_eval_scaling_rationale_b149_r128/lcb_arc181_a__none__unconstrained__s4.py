import sys

def solve():
    # Read all input at once and split into a list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # T is the first element
    T = int(input_data[0])
    
    # We need to group the remaining tokens into test cases.
    # Since each case starts with N followed by N elements, 
    # we can't use a simple slice. However, we can use a 
    # generator or a custom function to parse the flat list.
    
    def get_cases(data):
        # Using a generator to yield (N, P) pairs
        # We use a helper list to track the current index
        idx = [1]
        def next_case():
            if idx[0] >= len(data):
                return None
            n = int(data[idx[0]])
            p = data[idx[0]+1 : idx[0]+1+n]
            idx[0] += 1 + n
            return (n, p)
        
        # To avoid a loop, we can use map with a range and a 
        # closure, but since we need to track the index, 
        # a recursive-like structure via map is tricky.
        # Instead, we calculate the offsets first.
        
        # Calculate the starting position of each test case
        # offsets[i] is the start of case i
        # This is the only place where we "iterate", but we do it 
        # via a comprehension to find the boundaries.
        
        # Because N varies, we must determine the boundaries.
        # Let's use a different approach: 
        # We know the total sum of N is 2*10^5.
        # We can use a while-loop inside a function, but the 
        # constraint says no while/for loops.
        # Let's use a recursive-like structure using map and a 
        # mutable state container.
        
        return next_case

    # Since we cannot use loops, we use map() to process the cases.
    # But wait, the get_cases logic above still needs a way to 
    # call next_case T times.
    
    # Correct approach to handle variable N without loops:
    # 1. Read T.
    # 2. Use a helper function with a mutable index to extract cases.
    # 3. Use map(range(T)) to trigger the extraction.
    
    cursor = [1]
    def extract( _ ):
        n = int(input_data[cursor[0]])
        p = list(map(int, input_data[cursor[0]+1 : cursor[0]+1+n]))
        cursor[0] += 1 + n
        return n, p

    # Process each case
    # The logic for the problem:
    # 0 ops: if P is already sorted.
    # 1 op: if there exists k such that sorting [1, k-1] and [k+1, N] 
    #       results in [1, ..., N].
    #       This happens if P_k = k and all elements < k are in 
    #       positions 1...k-1 (in any order) and all elements > k 
    #       are in positions k+1...N (in any order).
    #       Actually, the condition is simpler: 
    #       If we pick k, the result is sorted if and only if 
    #       {P_1, ..., P_{k-1}} = {1, ..., k-1} and 
    #       {P_{k+1}, ..., P_N} = {k+1, ..., N}.
    #       This implies P_k must be k.
    # 2 ops: Otherwise, it's always possible in 2 ops (given N >= 3).
    
    def calculate_min_ops(case):
        n, p = case
        # Check if already sorted
        # We use all() which is allowed as it's a builtin
        if all(p[i] == i + 1 for i in range(n)):
            return 0
        
        # Check if 1 operation is enough
        # We need to find k such that P_k = k and 
        # max(P_1...P_{k-1}) < k and min(P_{k+1}...P_N) > k.
        # This is equivalent to saying:
        # For a fixed k, if P_k = k, we check if the set of elements
        # to the left is {1...k-1}.
        
        # To do this efficiently without loops, we use prefix max and suffix min.
        # Since we can't use loops, we use list comprehensions.
        # But wait, prefix max requires the previous value. 
        # We can use itertools.accumulate.
        from itertools import accumulate
        
        pref_max = list(accumulate(p, max))
        # For suffix min, we reverse, accumulate, then reverse back.
        suff_min = list(accumulate(p[::-1], min))[::-1]
        
        # k is 1-indexed in the problem, so index i = k-1.
        # Condition for k:
        # If i == 0: P_0 == 1 is not enough, we need suff_min[1] > 1 (if N>1)
        # If i == n-1: P_{n-1} == n and pref_max[n-2] < n
        # If 0 < i < n-1: P_i == i+1 and pref_max[i-1] < i+1 and suff_min[i+1] > i+1
        
        # We can check this for all i using a list comprehension and any()
        # Handle boundaries carefully.
        
        def check_k(i):
            if p[i] != i + 1: return False
            left_ok = (i == 0) or (pref_max[i-1] < i + 1)
            right_ok = (i == n - 1) or (suff_min[i+1] > i + 1)
            return left_ok and right_ok

        if any(check_k(i) for i in range(n)):
            return 1
        
        return 2

    # Use map to apply the logic to all cases
    results = map(calculate_min_ops, map(extract, range(T)))
    
    # Print all results separated by newlines
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()