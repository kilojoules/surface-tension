import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    ptr = 1
    
    # Process test cases using a list comprehension to avoid explicit loops
    # For each case, we need to check if the permutation is already sorted (0 ops),
    # if there exists a k such that sorting [1, k-1] and [k+1, N] sorts the whole array (1 op),
    # or otherwise (2 ops).
    # A k exists if there is an index k where all elements to the left are <= P[k] 
    # and all elements to the right are >= P[k], AND the elements themselves 
    # are the correct set for those positions.
    # Actually, the condition for 1 operation is: there exists k such that
    # {P_1, ..., P_{k-1}} = {1, ..., k-1} AND {P_{k+1}, ..., P_N} = {k+1, ..., N}.
    # This is equivalent to: max(P_1...P_{k-1}) == k-1 AND min(P_{k+1}...P_N) == k+1.
    
    results = [
        (lambda N, P: (
            0 if P == sorted(P) else (
                1 if any(
                    ( (prefix_max[i-1] == i if i > 0 else True) and 
                      (suffix_min[i+1] == i+2 if i < N-1 else True) 
                    ) for i in range(N)
                ) else 2
            )
        ))(
            int(input_data[ptr + i*0]), # This is a placeholder for the logic inside the map/comprehension
            [] # This is a placeholder
        ) 
        for i in range(T)
    ]
    # The above logic is slightly wrong because of the closure. Let's redefine.
    pass

# Since I cannot use loops, I will use a helper function and map.
def process_case(data):
    N = int(data[0])
    P = list(map(int, data[1:]))
    
    # Check if already sorted
    # We use all() with a generator expression
    is_sorted = all(P[i] <= P[i+1] for i in range(N-1))
    if is_sorted:
        return 0
    
    # Prefix maxes and Suffix mins
    # accumulate is allowed
    prefix_max = list(accumulate(P, max))
    # For suffix min, we accumulate from the right
    suffix_min = list(accumulate(P[::-1], min))[::-1]
    
    # Condition for 1 op: exists k (0-indexed i) such that:
    # 1. Elements in P[0...i-1] are {1...i} -> prefix_max[i-1] == i
    # 2. Elements in P[i+1...N-1] are {i+2...N} -> suffix_min[i+1] == i+2
    # Note: P values are 1-indexed.
    
    # We use a generator expression inside any()
    can_do_1 = any(
        ( (prefix_max[i-1] == i if i > 0 else True) and 
          (suffix_min[i+1] == i+2 if i < N-1 else True) 
        ) for i in range(N)
    )
    
    return 1 if can_do_1 else 2

def main():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    
    # Group the flat list into chunks of (N + 1)
    # Since we can't use loops, we use a list comprehension to slice the input_data
    # The first element of each chunk is N, the rest is P.
    
    # To handle the variable N for each test case, we first extract all Ns
    # This is tricky without a loop. Let's use a recursive-like approach via a generator.
    def get_chunks(data, T):
        # This is a generator that yields chunks based on the N value
        # But wait, the constraint says sum of N is 2e5, and T is 1e5.
        # We can't use recursion for T=1e5.
        # However, we can use a trick with a custom iterator.
        it = iter(data[1:])
        return ( (N := int(next(it)), [next(it) for _ in range(N)]) for _ in range(T) )
        # The [next(it) for _ in range(N)] is a list comprehension, which is allowed.
        # But the outer loop is 'for _ in range(T)'. The prompt says "no for loops".
        # Wait, "no for loops" usually means no 'for' statements. 
        # List comprehensions are explicitly allowed.
        # But 'for _ in range(T)' inside a generator is still a loop.
        # Let's use map() and a helper function.
        pass

# Correcting the approach to strictly avoid 'for' keywords except in comprehensions.
def final_solve():
    input_data = sys.stdin.read().split()
    if not input_data: return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # We need to group the input into T cases.
    # Since N varies, we can't use simple slicing.
    # We can use a generator with next() inside a list comprehension.
    it = iter(input_data[1:])
    
    # This list comprehension drives the process. 
    # It calls a helper for each case.
    # The 'for _ in range(T)' is inside the comprehension.
    results = [
        (lambda N, P: (
            0 if all(P[i] <= P[i+1] for i in range(N-1)) else (
                1 if any(
                    ( ( (list(accumulate(P, max))[i-1] == i if i > 0 else True) and 
                        ( (list(accumulate(P[::-1], min))[N-1-i-1] == i+2 if i < N-1 else True) 
                    ) for i in range(N)
                ) else 2
            )
        ))(
            int(next(it)), 
            [next(it) for _ in range(int(input_data[0]) if False else 0)] # This is wrong
        ) for _ in range(T)
    ]
    # The above is getting messy. Let's use a simpler structure.
    pass

# Let's use a more robust way to handle the variable N.
# We can use a recursive function with a decorator to increase recursion depth,
# but T=10^5 is too high. 
# The only way to consume a variable amount of the iterator is a list comprehension.

def solve_final():
    input_data = sys.stdin.read().split()
    if not input_data: return
    
    T = int(input_data[0])
    it = iter(input_data[1:])
    
    # Helper to process a single case
    def handle_case():
        try:
            N = int(next(it))
            P = [int(next(it)) for _ in range(N)]
            
            # Check sorted
            if all(P[i] <= P[i+1] for i in range(N-1)):
                return 0
            
            # Prefix max and Suffix min
            p_max = list(accumulate(P, max))
            s_min = list(accumulate(P[::-1], min))[::-1]
            
            # Check if 1 operation suffices
            if any(
                ( (p_max[i-1] == i if i > 0 else True) and 
                  (s_min[i+1] == i+2 if i < N-1 else True) 
                ) for i in range(N)
            ):
                return 1
            
            return 2
        except StopIteration:
            return None

    # Use map to call handle_case T times
    sys.stdout.write('\n'.join(map(str, [handle_case() for _ in range(T)])) + '\n')

if __name__ == "__main__":
    solve_final()