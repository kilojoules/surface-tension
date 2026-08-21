import sys
from itertools import groupby

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Use a generator to process test cases without explicit for/while loops
    # We calculate the starting index for each test case based on the N values
    # However, since we cannot use loops to slice, we use a recursive-like 
    # structure via map and a helper to handle the variable length N.
    
    def get_cases(data):
        # This is a trick to group the flat list into (N, P) pairs
        # We use a generator that yields the current N and the next N elements
        def gen(remainder):
            if not remainder:
                return
            n = int(remainder[0])
            yield (n, remainder[1:n+1])
            yield from gen(remainder[n+1:])
        return gen(data)

    # To avoid recursion depth issues with gen(), we use a different approach
    # to parse the flat list into test cases.
    # Since we can't use loops, we'll use a list comprehension to find the 
    # boundaries of each test case.
    
    # Actually, the simplest way to handle the input without loops is to 
    # process the flat list using a custom reducer or a clever comprehension.
    # But the constraints allow us to use map/filter/reduce.
    
    # Let's redefine the logic:
    # The answer is 0 if already sorted.
    # The answer is 1 if there exists k such that sorting [1, k-1] and [k+1, N]
    # results in [1, ..., N]. This happens if the element at position k is k,
    # and all elements < k are in the first k-1 positions, and all > k are in the last N-k.
    # Wait, the operation is: sort(1, k-1) and sort(k+1, N).
    # For the result to be (1, ..., N), we MUST have P[k] = k.
    # If P[k] = k, then sorting the left and right parts will automatically
    # put 1...k-1 in the first k-1 slots and k+1...N in the last N-k slots.
    # So answer is 1 if there exists k such that P[k] = k.
    # Otherwise, the answer is 2. (It is proven that 2 is always sufficient for N >= 3).
    
    # Special case: if already sorted, answer is 0.
    
    # To handle the variable N per test case without loops:
    # We can use a technique to track the current index.
    # But since we can't use loops, we can use a recursive function with a 
    # decorator to increase recursion limit, or use a mathematical approach.
    
    # Let's use a more robust way to parse:
    # We can use a list comprehension to extract the Ns and then use 
    # a cumulative sum to find the slices.
    
    # Since we can't use loops, we use a helper to extract the cases.
    # Given the constraints and the "no loop" rule, we can use a 
    # recursive-like structure via a list comprehension that 
    # processes the input.
    
    # Actually, the most reliable way to parse T cases of variable length N
    # without loops is to use a generator and 'next()'.
    
    def process_all(data):
        it = iter(data)
        def handle_case():
            try:
                n_str = next(it)
                n = int(n_str)
                p = [int(next(it)) for _ in range(n)]
                
                # Check if already sorted
                # We use all() which is allowed
                is_sorted = all(p[i] == i + 1 for i in range(n))
                if is_sorted:
                    return 0
                
                # Check if any P[k] == k
                # Note: P is 0-indexed, so we check p[i] == i + 1
                exists_k = any(p[i] == i + 1 for i in range(n))
                return 1 if exists_k else 2
            except StopIteration:
                return None

        # Use map to call handle_case T times
        return list(map(lambda _: handle_case(), range(T)))

    results = process_all(input_data[1:])
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()