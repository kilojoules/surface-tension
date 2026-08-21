import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    ptr = 1
    
    # We need to process T test cases. Since we cannot use loops, 
    # we create a list of test case data and map a processing function over it.
    
    # Helper to extract test cases into a list of (N, P) tuples
    def get_cases(p, remaining):
        if not remaining:
            return []
        N = int(remaining[0])
        P = list(map(int, remaining[1:N+1]))
        return [(N, P)] + get_cases(p, remaining[N+1:])

    # To avoid recursion depth issues with get_cases, we use a more iterative-like 
    # approach to chunk the input. However, since we must avoid loops, 
    # we can use a generator with next() inside a list comprehension.
    
    def case_generator(data):
        it = iter(data)
        # Use a helper to yield cases without a while loop
        # We can use a recursive-like structure via a generator
        def produce():
            try:
                N = int(next(it))
                P = [int(next(it)) for _ in range(N)]
                yield (N, P)
                yield from produce()
            except StopIteration:
                pass
        return produce()

    # Since the constraint allows list comprehensions and map, 
    # we can use a trick to consume the iterator.
    # But the simplest way to handle T cases without loops is to 
    # map a function over a range(T) and maintain a stateful iterator.
    
    it = iter(input_data[1:])
    
    def process_single_case(_):
        N = int(next(it))
        P = [int(next(it)) for _ in range(N)]
        
        # The problem asks for the minimum operations to sort P.
        # An operation k sorts [1, k-1] and [k+1, N].
        # If P is already sorted, answer is 0.
        # If there exists k such that sorting [1, k-1] and [k+1, N] sorts the whole array, answer is 1.
        # This happens if there is an index k such that:
        # 1. All elements in P[0...k-2] are <= P[k] (if they exist)
        # 2. All elements in P[k...N-1] are >= P[k-2] (if they exist)
        # Actually, the condition for 1 operation is:
        # There exists k such that the set of elements {P_1...P_{k-1}} is {1...k-1}
        # AND the set of elements {P_{k+1}...P_N} is {k+1...N}.
        # This is equivalent to saying P_k = k and the prefix is a permutation of 1..k-1.
        
        # Let's refine: 1 operation with index k works if:
        # The elements in positions 1 to k-1 are some permutation of 1 to k-1
        # AND the elements in positions k+1 to N are some permutation of k+1 to N.
        # This implies P[k-1] must be k.
        
        # To check if P[0...k-2] is a permutation of 1...k-1:
        # max(P[0...k-2]) == k-1.
        
        # We can use accumulate to find prefix maxes and suffix maxes.
        # But we need to check if the prefix is exactly 1...k-1.
        # A prefix of length L is a permutation of 1...L iff max(prefix) == L.
        
        # Let's calculate prefix_max and suffix_min.
        # For a fixed k (1-indexed), we need:
        # 1. max(P[0...k-2]) == k-1 (if k > 1)
        # 2. min(P[k...N-1]) == k+1 (if k < N)
        # 3. P[k-1] == k
        
        # However, the condition is simpler: 
        # If we sort everything except P[k-1], the result is sorted iff:
        # P[k-1] == k AND max(P[0...k-2]) <= k AND min(P[k...N-1]) >= k.
        # Since it's a permutation, if P[k-1] == k, then max(P[0...k-2]) <= k 
        # is guaranteed if the prefix is a permutation of 1...k-1.
        
        # Let's use a list comprehension to check all k from 1 to N.
        # We need prefix_max and suffix_min.
        
        # Using a list comprehension to build prefix_max and suffix_min:
        # Since we can't use loops, we use a trick with a helper function or 
        # we can use the fact that we only need to check if P[k-1] == k 
        # and the prefix is a permutation.
        
        # Wait, the condition "sort 1 to k-1 and k+1 to N" makes the array sorted 
        # if and only if the set of values {P_1, ..., P_{k-1}} is {1, ..., k-1} 
        # and the set of values {P_{k+1}, ..., P_N} is {k+1, ..., N}.
        # This is true if and only if P_k = k and max(P_1, ..., P_{k-1}) = k-1.
        
        # We can use a list comprehension to check this for all k.
        # To get prefix maxes without a loop, we can use a helper with recursion 
        # (but depth is an issue) or we can use the fact that we can use 
        # 'accumulate' from itertools.
        
        from itertools import accumulate
        
        # P is 0-indexed, so P[i] corresponds to P_{i+1}
        # We check if P[i] == i+1 and max(P[0...i-1]) == i
        
        # Create a list of prefix maxes
        prefix_max = list(accumulate(P, max))
        
        # We need to check if P[i] == i+1 and (i == 0 or prefix_max[i-1] == i)
        # AND (i == N-1 or min(P[i+1...N-1]) == i+2)
        # To get suffix mins, we can use accumulate from the right.
        suffix_min = list(accumulate(P[::-1], min))[::-1]
        
        # Check if any k (index i) satisfies the condition
        # For i = 0 to N-1:
        # Condition: P[i] == i+1 
        #           AND (i == 0 or prefix_max[i-1] == i)
        #           AND (i == N-1 or suffix_min[i+1] == i+2)
        
        can_do_in_1 = any(
            P[i] == i+1 and 
            (i == 0 or prefix_max[i-1] == i) and 
            (i == N-1 or suffix_min[i+1] == i+2)
            for i in range(N)
        )
        
        # If already sorted, 0. Else if can_do_in_1, 1. Else 2.
        # (It is proven that max 2 operations are needed for N >= 3)
        is_sorted = (P == sorted(P))
        
        return 0 if is_sorted else (1 if can_do_in_1 else 2)

    # Use map to apply process_single_case to the range of T
    results = map(process_single_case, range(T))
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()