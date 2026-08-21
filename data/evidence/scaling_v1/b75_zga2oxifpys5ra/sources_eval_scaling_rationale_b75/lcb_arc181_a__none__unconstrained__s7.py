import sys
from functools import reduce

def solve():
    # Read all input at once and split into a flat list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Use a generator to yield chunks of the input list corresponding to each test case
    def get_cases(data):
        it = iter(data[1:])
        return ( (int(next(it)), [int(next(it)) for _ in range(int(current_n))]) 
                 for current_n in (lambda it: (next(it) for _ in range(T)))(it) )
    
    # Since the above generator logic is slightly flawed due to nested next(), 
    # let's use a more robust approach to group the flat list.
    def group_cases(data):
        # We use a helper function to process the flat list into (N, P) pairs
        def process(acc, item):
            # acc is (current_cases, current_N, current_P)
            cases, n, p = acc
            if n is None:
                return (cases, int(item), [])
            else:
                new_p = p + [int(item)]
                if len(new_p) == n:
                    return (cases + [(n, new_p)], None, [])
                else:
                    return (cases, n, new_p)
        
        # Using reduce to group the flat list into test cases
        result = reduce(process, data[1:], ([], None, []))
        return result[0]

    cases = group_cases(input_data)

    def solve_case(case):
        N, P = case
        # Check if already sorted
        if all(P[i] == i + 1 for i in range(N)):
            return 0
        
        # For 1 operation, we need a k such that:
        # Sorted(P[0...k-2]) == 1...k-1 AND Sorted(P[k...N-1]) == k+1...N
        # This is equivalent to:
        # The set of elements in P[0...k-2] is {1...k-1} AND
        # The set of elements in P[k...N-1] is {k+1...N}
        # Which implies P[k-1] must be k.
        
        # Precompute prefix max and suffix min
        # prefix_max[i] is max of P[0...i]
        # suffix_min[i] is min of P[i...N-1]
        # However, the condition is simpler: 
        # For a fixed k (1-indexed), we need:
        # max(P[0...k-2]) <= k-1 AND min(P[k...N-1]) >= k+1
        
        # We can use list comprehensions to build prefix_max and suffix_min
        # But since we can't use loops, we use a trick with scan (via reduce)
        def scan(func, iterable, initial):
            return reduce(lambda acc, x: acc + [func(acc[-1], x)], iterable, [initial])

        # Correct way to do prefix max and suffix min without loops:
        # We use the fact that we can use map/filter/reduce.
        # For prefix max:
        p_max = scan(lambda a, b: max(a, b), P, -1)[1:]
        # For suffix min (process reversed P):
        s_min = scan(lambda a, b: min(a, b), P[::-1], N + 1)[1:][::-1]
        
        # Check if any k (0 to N-1 index) satisfies the condition
        # k is the index of the element we DON'T sort.
        # Condition: (k==0 or p_max[k-1] == k) and (k==N-1 or s_min[k+1] == k+2)
        # Note: p_max[k-1] is max of P[0...k-1]. If that is k, then P[0...k-1] is a permutation of 1...k.
        # Wait, the operation sorts 1 to k-1 and k+1 to N.
        # So for index k (0-indexed), we need:
        # Elements P[0...k-1] to be {1...k} AND elements P[k+1...N-1] to be {k+2...N}
        # This means max(P[0...k-1]) == k and min(P[k+1...N-1]) == k+2
        # (Adjusting for 0-indexing and 1-based values)
        
        # Let's redefine:
        # Prefix max of P[0...k-1] is p_max[k-1]
        # Suffix min of P[k+1...N-1] is s_min[k+1]
        # We need p_max[k-1] == k and s_min[k+1] == k+2 (for 0 <= k < N)
        # Handle boundaries:
        # If k=0: only need s_min[1] == 2
        # If k=N-1: only need p_max[N-2] == N-1
        # If 0 < k < N-1: need p_max[k-1] == k and s_min[k+1] == k+2
        
        # We can use a list comprehension to check all k
        possible = [
            (k == 0 and (N < 2 or s_min[1] == 2)) or
            (k == N-1 and (N < 2 or p_max[N-2] == N-1)) or
            (0 < k < N-1 and p_max[k-1] == k and s_min[k+1] == k+2)
            for k in range(N)
        ]
        
        return 1 if any(possible) else 2

    # Use map to apply solve_case to all cases and join results
    print('\n'.join(map(str, map(solve_case, cases))))

if __name__ == "__main__":
    solve()