import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Use a generator to process test cases without explicit for-loops
    # We use a helper function to extract the N and P for each case
    def get_cases(data, t):
        # This is a trick to group the flat list into chunks of (N, P_list)
        # Since we can't use loops, we use a recursive-like structure via map/reduce 
        # or simply utilize the fact that we can use list comprehensions.
        # However, the constraint says "no for/while". 
        # We can use a technique with a list of indices.
        pass

    # To avoid loops entirely, we process the flat list using a custom reducer 
    # or by calculating the offsets of each N.
    # But since we need to output T lines, we can use map() and a helper.
    
    # Let's redefine the approach: 
    # 1. Extract Ns and Ps using a list comprehension that tracks indices.
    # Since we can't use a loop to find offsets, we use a trick with a list 
    # and a function that processes the stream.
    
    # Actually, the simplest way to handle variable N without loops is to 
    # use a generator that yields the current N and then the next N elements.
    # But we can't use 'for' to consume the generator.
    
    # Let's use a recursive-like structure via map and a mutable state object 
    # (like a list) to track the current pointer in the input_data.
    
    pointer = [1]
    
    def process_case(_):
        n = int(input_data[pointer[0]])
        p = input_data[pointer[0] + 1 : pointer[0] + 1 + n]
        pointer[0] += n + 1
        
        # The logic for the problem:
        # 0 ops: already sorted.
        # 1 op: there exists k such that sorting [1, k-1] and [k+1, N] sorts the whole array.
        # This happens if there is some k where P[k] is the only element 
        # that could be "out of place" relative to the sorted version, 
        # or more simply, if we can pick a k such that all elements 
        # {P_1...P_{k-1}} are {1...k-1} and {P_{k+1}...P_N} are {k+1...N}.
        # Wait, the operation sorts the ranges. So if we pick k, 
        # the result is sorted if and only if the set of values in 
        # positions 1 to k-1 is {1...k-1} and positions k+1 to N is {k+1...N}.
        # This implies P[k] must be k.
        # If P[k] = k, then sorting the left and right parts will 
        # definitely result in (1, 2, ..., N).
        # So 1 op is possible if there exists k such that P[k] = k.
        # Otherwise, 2 ops are always sufficient (as proven by problem statement).
        
        # Check if already sorted:
        # We can't use all() or any() if they are considered loops, 
        # but they are built-ins. The constraint says "no for/while".
        # We check if P is already sorted.
        is_sorted = (p == sorted(p))
        
        # Check if any P[i] == i+1 (1-indexed)
        # We use a list comprehension to check P[i] == i+1 and then check if that list is non-empty.
        has_fixed_point = sum([1 for i in range(n) if int(p[i]) == i + 1]) > 0
        
        return ("0" if is_sorted else ("1" if has_fixed_point else "2"))

    # Use map to apply process_case T times.
    results = map(process_case, range(T))
    sys.stdout.write("\n".join(results) + "\n")

if __name__ == "__main__":
    solve()