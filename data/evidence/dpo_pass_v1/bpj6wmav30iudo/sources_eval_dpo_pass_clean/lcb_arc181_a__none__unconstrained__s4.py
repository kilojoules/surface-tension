import sys

def solve():
    # Read all input at once and split into a flat list of strings
    data = sys.stdin.read().split()
    if not data:
        return
    
    # T is the first element
    T_str = data[0]
    T = int(T_str)
    
    # We process the remaining data. Since N varies, we can't use a simple 
    # slice. We use an iterator to consume the data list.
    it = iter(data[1:])
    
    # Use a list comprehension to process each test case.
    # For each case:
    # 1. Read N.
    # 2. Read P as a list of integers.
    # 3. Check if sorted: 0.
    # 4. Check if there exists k such that P[k-1] == k and 
    #    max(P[0...k-2]) == k-1 and min(P[k...N-1]) == k+1: 1.
    # 5. Otherwise: 2.
    # To avoid loops, we map a lambda over a range(T).
    
    # Since we cannot use loops, we must handle the iterator carefully.
    # We can't use a loop to call next(it), but we can use a 
    # recursive-like structure via map/list comprehension 
    # if we can index the data.
    
    # Let's redefine the approach to use indexing on the flat list 'data'.
    # We need to track the current position in 'data'.
    # Since we can't use loops, we'll pre-calculate the start indices of each case.
    
    # First, we find the indices where each N is located.
    # This is tricky without loops. Let's use a different strategy.
    # We can use a generator or map with a helper that manages state.
    # But the prompt says "no for/while loops". 
    # We can use 'itertools.accumulate' or similar, but that's complex.
    # Actually, we can process the flat list by calculating 
    # the boundaries of each test case.
    
    # Let's use a recursive-like approach via map and a shared list/it.
    # Wait, the prompt allows list comprehensions. 
    # I can use a helper function and map.
    
    pass

# Because of the "no loop" constraint, I will use a recursive-style 
# processing via map and an iterator. 
# Note: The prompt says "no recursion", but map(lambda...) is allowed.

def main():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    
    # To handle variable N without loops, we can use a generator 
    # that yields the slices of the input_data.
    # But we can't use a loop to build that generator.
    # However, we can use a recursive-like structure with map 
    # if we pass the iterator.
    
    it = iter(input_data[1:])
    
    # This function processes a single test case
    process_case = lambda: (
        lambda N, P: (
            0 if P == sorted(P) else (
                1 if any(
                    (P[k-1] == k and 
                     (k == 1 or max(P[:k-1]) == k-1) and 
                     (k == len(P) or min(P[k:]) == k+1))
                    for k in range(1, len(P) + 1)
                ) else 2
            )
        )
    )(int(next(it)), [int(next(it)) for _ in range(int(next(it) if False else 0))]) 
    # The above is a placeholder. Since we can't use loops or recursion,
    # and N is variable, the only way to consume the iterator 
    # is via map/list comprehension.
    
    # Correct approach for variable N without loops:
    # Use a list comprehension that consumes the iterator.
    # Since we need N to determine how many P's to read, 
    # we can use a trick: 
    # we read N, then we read the next N elements.
    
    # Let',s use a helper function that gets called T times.
    # To avoid the loop, we use map(lambda _, __: ..., range(T)).
    
    # We need a way to read N and then N elements.
    # We can use [next(it) for _ in range(N)] inside the lambda.
    
    # Define the logic in a lambda
    logic = lambda: (
        lambda N: (
            lambda P: (
                0 if P == sorted(P) else (
                    1 if any(
                        (P[k-1] == k and 
                         (k == 1 or max(P[:k-1]) == k-1) and 
                         (k == N or min(P[k:]) == k+1))
                        for k in range(1, N + 1)
                    ) else 2
                )
            )([int(next(it)) for _ in range(N)])
        )(int(next(it)))
    )
    
    # Execute T times and print
    print('\n'.join(map(lambda _: logic(), range(T))))

if __name__ == "__main__":
    main()