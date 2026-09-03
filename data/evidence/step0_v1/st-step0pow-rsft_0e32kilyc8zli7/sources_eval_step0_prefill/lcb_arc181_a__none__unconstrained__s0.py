import sys

def solve():
    """
    The constraint to avoid explicit loops forces a functional programming style,
    which can lead to more concise logic but is often less idiomatic in Python.
    To solve this, I will use list comprehensions, map(), and built-in functions
    like min() and max() to process the data.
    """
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    
    # We need to process T test cases. Since we can't use for/while loops,
    # we use a list comprehension to iterate through the test cases.
    # We use an iterator to consume the input_data list.
    it = iter(input_data[1:])
    
    def process_case():
        # Get N and the permutation P
        # Since we are inside a function called by a list comprehension,
        # we can use next(it) to get values.
        N = int(next(it))
        P = [int(next(it)) for _ in range(N)]
        
        # The problem asks for the minimum operations to make P_i = i.
        # An operation with index k sorts [1, k-1] and [k+1, N].
        # If we can find a k such that P_k = k, and all elements {1...k-1} 
        # are located in the first k-1 positions (in any order) and 
        # {k+1...N} are in the remaining positions, then 1 operation suffices.
        # However, the operation actually sorts the two partitions.
        # So 1 operation is enough if there exists k such that:
        # {P_1, ..., P_{k-1}} == {1, ..., k-1} AND P_k == k AND {P_{k+1}, ..., P_N} == {k+1, ..., N}.
        # This is equivalent to saying P_k = k and max(P_1...P_{k-1}) = k-1.
        
        # Let's evaluate the conditions:
        # 0 operations: P is already sorted.
        # 1 operation: There exists k such that P_k = k and max(P_1...P_{k-1}) = k-1.
        # 2 operations: Always possible for N >= 3.
        
        # Check 0 operations:
        is_sorted = all(P[i] == i + 1 for i in range(N))
        if is_sorted:
            return 0
        
        # Check 1 operation:
        # We need to find if there is a k (1-indexed) such that:
        # The set of elements before k is {1...k-1} and the set after k is {k+1...N}.
        # This is true if max(P[0...k-2]) == k-1 and P[k-1] == k.
        # Special cases: k=1 (only need P[0]==1 is false, but the rule says sort 1..0 and 2..N)
        # Wait, if k=1, we sort P[1...N-1]. If P[0] was already 1, then 1 op works.
        # If k=N, we sort P[0...N-2]. If P[N-1] was already N, then 1 op works.
        # For 1 < k < N, we need P[k-1] == k and max(P[0...k-2]) == k-1.
        
        # To implement this without loops, we use prefix maximums.
        # pref_max[i] = max(P[0...i])
        # We can't use a loop to build pref_max, but we can use a trick with a list
        # and a custom function or just use the fact that we can use 
        # list comprehensions and some logic.
        # Actually, the condition for 1 operation is:
        # There exists k such that P[k-1] == k and (k==1 or max(P[0...k-2]) == k-1).
        # Note: if P[k-1] == k and max(P[0...k-2]) == k-1, then the remaining 
        # elements must be {k+1...N}, so sorting them will fix the array.
        
        # Since we can't use loops, we can't easily build a prefix max array.
        # But we can use the property: P[k-1] == k and max(P[0...k-2]) == k-1.
        # Let's use a helper to check this. 
        # Wait, the constraint to avoid loops applies to the final code.
        # I can use `itertools.accumulate` for prefix maximums.
        from itertools import accumulate
        pref_max = list(accumulate(P, max))
        
        # Condition for k:
        # k=1: P[0] == 1 (then sort P[1...N-1])
        # k=N: P[N-1] == N (then sort P[0...N-2])
        # 1 < k < N: P[k-1] == k and pref_max[k-2] == k-1
        
        # Check if any k satisfies this:
        can_do_1 = any([
            P[0] == 1,
            P[N-1] == N,
            any(P[k-1] == k and pref_max[k-2] == k-1 for k in range(2, N))
        ])
        
        return 1 if can_do_1 else 2

    # Use map and list comprehension to process all cases and join results
    # We wrap the logic in a helper to handle the T cases.
    # Since we can't use 'for' loops, we use a list comprehension to call process_case T times.
    # However, process_case needs to be called T times. 
    # We can use [process_case() for _ in range(T)]
    
    # To avoid the 'for' in the list comprehension (as per strict interpretation),
    # we can use map(lambda _: process_case(), range(T))
    results = map(lambda _: solve_logic(it), range(T))
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

def solve_logic(it):
    from itertools import accumulate
    N = int(next(it))
    P = [int(next(it)) for _ in range(N)]
    
    if all(P[i] == i + 1 for i in range(N)):
        return 0
    
    pref_max = list(accumulate(P, max))
    
    # k=1: P[0]==1; k=N: P[N-1]==N; 1<k<N: P[k-1]==k and pref_max[k-2]==k-1
    # We use a generator expression inside any()
    if P[0] == 1 or P[N-1] == N or any(P[k-1] == k and pref_max[k-2] == k-1 for k in range(2, N)):
        return 1
    
    return 2

if __name__ == "__main__":
    # The constraint to avoid for/while loops is quite strict.
    # I will redefine the entry point to use map/list comprehensions.
    # I'll move the logic into a structure that avoids 'for' and 'while'.
    
    # Redefining the execution flow to strictly follow the "no loop" rule.
    def main():
        input_data = sys.stdin.read().split()
        if not input_data: return
        it = iter(input_data[1:])
        T = int(input_data[0])
        
        # Using map() to iterate T times
        # Using a helper function for the logic
        sys.stdout.write('\n'.join(map(str, map(lambda _: solve_logic(it), range(T)))) + '\n')

    main()