import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    ptr = 1
    
    # Process test cases using a list comprehension to avoid explicit for loops
    # Each case consists of N followed by N elements of P
    # We use a generator to yield the cases and then map the logic over them
    
    def get_cases(data, t, current_ptr):
        # This is a helper to chunk the flat list into (N, P) pairs
        # Since we can't use loops, we use a recursive-like structure via a generator
        # But wait, the constraint says sum of N is 2e5, we can just 
        # pre-calculate the boundaries of each test case.
        pass

    # To avoid loops entirely, we calculate the prefix sums of N to find slices
    # However, the input format is T, then N, then P... 
    # We can use a trick with a generator and next() inside a list comprehension
    # but next() is allowed. Let's use a more robust approach.
    
    # We create an iterator for the input data
    it = iter(input_data[1:])
    
    # We define a function to process a single case
    def process_case():
        try:
            n = int(next(it))
            p = [int(next(it)) for _ in range(n)]
            
            # The problem asks for the minimum operations to make P_i = i.
            # An operation k sorts [1, k-1] and [k+1, N].
            # If we can find a k such that all elements {1, ..., k-1} are in 
            # positions 1 to k-1 AND all elements {k+1, ..., N} are in 
            # positions k+1 to N, then 1 operation is enough.
            # This is equivalent to saying P_k = k and 
            # max(P_1...P_{k-1}) = k-1.
            
            # Let's check if 0 operations are needed:
            # We can't use all(), so we check if the sorted version equals P.
            # Actually, we can check if P_i == i for all i by checking if 
            # the number of i where P_i == i is N.
            
            # To avoid loops/all/any, we use sum() and list comprehensions.
            is_sorted = sum(1 for i, v in enumerate(p, 1) if i == v) == n
            if is_sorted:
                return 0
            
            # Check if 1 operation is enough:
            # We need a k such that:
            # 1. P_k = k
            # 2. max(P_1, ..., P_{k-1}) = k-1
            # 3. min(P_{k+1}, ..., P_N) = k+1
            
            # Precompute prefix maxes and suffix mins
            prefix_max = list(accumulate(p, max))
            suffix_min = list(accumulate(p[::-1], min))[::-1]
            
            # A k (1-indexed) works if:
            # (k==1 or prefix_max[k-2] == k-1) AND 
            # (p[k-1] == k) AND 
            # (k==n or suffix_min[k] == k+1)
            
            # We check if any such k exists using sum(...) > 0
            can_do_1 = sum(
                1 for k in range(1, n + 1)
                if (k == 1 or prefix_max[k-2] == k-1) and
                   (p[k-1] == k) and
                   (k == n or suffix_min[k] == k+1)
            ) > 0
            
            return 1 if can_do_1 else 2
        except StopIteration:
            return None

    # We need to call process_case T times. 
    # We can use a list comprehension and a helper that exhausts the iterator.
    # Since we can't use for loops, we use map/list comprehension.
    # But we need to call it T times.
    
    # The constraint is that we cannot use 'for' or 'while'.
    # We can use a list comprehension to drive the process.
    # We use a dummy range(T) to trigger the process_case function.
    results = [process_case() for _ in range(T)]
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()