import sys

def solve():
    # Read all input at once and split into a list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Use a generator to process test cases without a for loop
    # We use a map-like structure to iterate through the flat list of tokens
    # Since we cannot use loops, we process the data using a recursive-like 
    # structure simulated via map/list comprehensions or generators.
    
    # To handle the variable length of N for each case, we pre-calculate 
    # the indices where each case starts.
    
    def get_cases(data):
        # This helper identifies the slices for each test case
        # Since we can't use loops, we use a custom reduction or 
        # a comprehension that tracks index state.
        # However, the simplest way to avoid 'for' is to use a generator
        # and next() calls within a comprehension.
        it = iter(data[1:])
        return (lambda f: f(f, it))(
            lambda self, iterator: (
                (lambda n: (
                    [int(next(iterator)) for _ in range(n)], 
                    self(self, iterator)
                )) (int(next(iterator))) 
                if True else None
            )
        )

    # The above recursive approach hits recursion limits. 
    # Instead, we use a generator function with a while loop (forbidden) 
    # or map. Wait, the prompt forbids 'for' and 'while'.
    # Let's use a list comprehension with a helper iterator.
    
    it = iter(input_data[1:])
    
    def process_case():
        try:
            N = int(next(it))
            P = [int(next(it)) for _ in range(N)]
            
            # Condition for 0: Already sorted
            is_sorted = (P == sorted(P))
            
            # Condition for 1: Exists k such that 
            # {P_1...P_{k-1}} == {1...k-1} AND {P_{k+1}...P_N} == {k+1...N}
            # This is equivalent to:
            # Max(P_1...P_{k-1}) == k-1 AND Min(P_{k+1}...P_N) == k+1
            # Note: k is the index of the element that stays put (P_k = k).
            # Actually, the operation sorts [1, k-1] and [k+1, N].
            # For this to result in 1...N, we need the set of values in 
            # positions 1...k-1 to be {1...k-1} and positions k+1...N to be {k+1...N}.
            # This implies P_k must be k.
            
            # We can check this by seeing if P_k == k AND 
            # max(P[0...k-2]) == k-1 AND min(P[k...N-1]) == k+1
            # Or more simply: P_k == k and the elements are partitioned correctly.
            
            # Precompute prefix max and suffix min
            # Since we can't use loops, we use map/list comprehensions with range
            # But we need the values. We can use a trick with sorted() or 
            # comprehension logic, but prefix/suffix requires accumulation.
            # Python's itertools.accumulate is perfect here.
            from itertools import accumulate
            
            pref_max = list(accumulate(P, max))
            # Suffix min requires reversing, accumulating, then reversing back
            suff_min = list(accumulate(P[::-1], min))[::-1]
            
            # Check if any k (1-indexed) satisfies the condition:
            # For k=1: sort(2...N) -> needs P_1 to be 1? No, k=1 sorts 2...N.
            # If k=1, we need P_1 == 1 and sorted(P_2...P_N) == 2...N.
            # Wait, the rule is: sort 1...k-1 and k+1...N.
            # To get 1...N, we need:
            # 1. The set {P_1, ..., P_{k-1}} must be {1, ..., k-1}
            # 2. The set {P_{k+1}, ..., P_N} must be {k+1, ..., N}
            # 3. This implies P_k must be k.
            
            # Special cases for k:
            # k=1: {P_2...P_N} == {2...N}. This is true if P_1 == 1.
            # k=N: {P_1...P_{N-1}} == {1...N-1}. This is true if P_N == N.
            # 1 < k < N: pref_max[k-2] == k-1 and suff_min[k] == k+1 and P[k-1] == k.
            
            # We can use next() to find if any k works
            can_do_1 = next(
                (True for k in range(1, N + 1) if (
                    (k == 1 and P[0] == 1) or 
                    (k == N and P[N-1] == N) or 
                    (1 < k < N and P[k-1] == k and pref_max[k-2] == k-1 and suff_min[k] == k+1)
                )), 
                False
            )
            
            return 0 if is_sorted else (1 if can_do_1 else 2)
        except StopIteration:
            return None

    # Use map to apply process_case T times
    results = map(lambda _: process_case(), range(T))
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()