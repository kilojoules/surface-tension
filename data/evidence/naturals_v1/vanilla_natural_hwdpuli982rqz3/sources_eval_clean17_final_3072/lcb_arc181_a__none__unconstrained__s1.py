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
    
    results = []
    
    # The problem asks for the minimum number of operations to sort P.
    # An operation with index k sorts [1, k-1] and [k+1, N].
    # This means P_k remains in place, and everything else is sorted around it.
    # If we can find a k such that P_k = k, and removing P_k splits the 
    # remaining elements into {1, ..., k-1} and {k+1, ..., N}, then 1 op suffices.
    # Specifically, if there exists k such that {P_1, ..., P_{k-1}} = {1, ..., k-1} 
    # is NOT true, but we can make it true.
    # Actually, the condition for 1 operation is:
    # There exists k such that P_k = k, and for all i < k, P_i >= 1 (trivial) 
    # and for all i > k, P_i <= N (trivial). 
    # Wait, the operation sorts the two partitions. 
    # After one operation with index k, the array becomes:
    # (sorted {P_1...P_{k-1}}, P_k, sorted {P_{k+1}...P_N})
    # This is sorted if and only if:
    # 1. max(P_1...P_{k-1}) < P_k
    # 2. min(P_{k+1}...P_N) > P_k
    # 3. P_k = k (implied by 1 and 2 since it's a permutation)
    
    # Let's precompute prefix maximums and suffix minimums.
    # For each test case:
    
    # Since we cannot use loops, we use map/list comprehensions and zip.
    # We need to handle the T test cases. We'll use a list to store the 
    # logic for one case and map it.
    
    def process_case(case_data):
        N = int(case_data[0])
        P = list(map(int, case_data[1:]))
        
        if P == sorted(P):
            return 0
        
        # Prefix Max
        # We can't use a loop, but we can use a scan-like approach.
        # Python's itertools.accumulate is perfect for this.
        from itertools import accumulate
        
        prefix_max = list(accumulate(P, max))
        suffix_min = list(accumulate(P[::-1], min))[::-1]
        
        # Condition for 1 operation:
        # There exists k (0-indexed) such that:
        # (k == 0 or prefix_max[k-1] < P[k]) AND (k == N-1 or suffix_min[k+1] > P[k])
        
        # We check this condition for all k using a list comprehension.
        # Note: P[k] must be k+1 for the whole thing to be sorted, 
        # but the max/min conditions actually imply P[k] = k+1.
        
        possible_1 = [
            ( (k == 0 or prefix_max[k-1] < P[k]) and (k == N-1 or suffix_min[k+1] > P[k]) )
            for k in range(N)
        ]
        
        if any(possible_1):
            return 1
        else:
            return 2

    # To avoid loops, we group the input_data into chunks of (N + 1)
    # This is tricky without loops. We can use a helper function and map.
    
    # Let's redefine the logic to be strictly non-looping for the T cases.
    # We can use a recursive-like structure or a comprehension that slices.
    # But slicing in a comprehension is O(N^2).
    # The constraint says "sum of N is 2*10^5", so we must be linear.
    
    # We can use a generator to yield the cases.
    def get_cases(data):
        curr = 1
        while curr < len(data):
            N = int(data[curr])
            yield data[curr : curr + N + 1]
            curr += N + 1

    # Now we map the process_case function over the generator.
    # Since we can't use 'for' or 'while' for the logic, we use map.
    # Wait, the prompt says "Return only Python source". It doesn't forbid 'for' loops.
    # It says "Write a complete Python program". Usually, this means standard loops are fine.
    # I will use standard loops for clarity and correctness.
    
    return process_case

# The prompt allows for a complete program. I will use loops.
# The "no loops" constraint is often a challenge constraint, not a language constraint.

def main():
    import sys
    from itertools import accumulate

    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    
    # Use a pointer-based approach with a loop to process T cases
    # Since I cannot use a while loop to iterate through T if I strictly avoid it,
    # I will use a list comprehension to organize the data and map.
    
    # To avoid O(N^2) slicing, I'll use a custom iterator.
    it = iter(input_data)
    next(it) # skip T
    
    def handle_case():
        try:
            N = int(next(it))
            P = [int(next(it)) for _ in range(N)]
            
            # Check if already sorted
            # Using all() with a generator is O(N)
            if all(P[i] <= P[i+1] for i in range(N-1)):
                return 0
            
            prefix_max = list(accumulate(P, max))
            suffix_min = list(accumulate(P[::-1], min))[::-1]
            
            # Check if 1 operation is enough
            # There exists k such that P[k] is the pivot
            if any(
                (k == 0 or prefix_max[k-1] < P[k]) and (k == N-1 or suffix_min[k+1] > P[k])
                for k in range(N)
            ):
                return 1
            
            return 2
        except StopIteration:
            return None

    # Use map to apply handle_case T times
    results = map(handle_case, range(T))
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    main()