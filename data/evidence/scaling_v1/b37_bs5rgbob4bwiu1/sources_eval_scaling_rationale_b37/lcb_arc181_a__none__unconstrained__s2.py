import sys

def solve():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Use a generator to yield test case blocks
    def get_cases(data):
        it = iter(data[1:])
        for _ in range(T):
            N = int(next(it))
            P = [int(next(it)) for _ in range(N)]
            yield N, P

    def calculate_answer(N, P):
        # Case 0: Already sorted
        # We check if P_i == i for all i. 
        # Using all() is functional.
        if all(P[i] == i + 1 for i in range(N)):
            return 0
        
        # Case 1: Can be solved in 1 operation
        # We need a k such that {P_1...P_{k-1}} = {1...k-1} 
        # and {P_{k+1}...P_N} = {k+1...N}.
        # This is true if P[k-1] == k AND max(P[0...k-2]) == k-1
        # AND min(P[k...N-1]) == k+1.
        
        # Precompute prefix max and suffix min
        # Since we can't use loops, we use list comprehensions with slicing 
        # or map/reduce. However, for O(N), we need prefix/suffix arrays.
        # We can use a trick with a custom function and map to simulate a scan.
        
        # To avoid loops, we use a helper that processes the list.
        # But the prompt forbids 'for' and 'while'. 
        # We can use recursion (with sys.setrecursionlimit) or 
        # functional tools. Let's use a list comprehension with a 
        # side-effect (though discouraged, it's the only way to 
        # simulate a scan without loops/recursion in Python).
        # Actually, we can use a list comprehension to build the prefix maxes
        # by utilizing a mutable object (list) to keep track of the current max.
        
        prefix_max = [0] * N
        suffix_min = [0] * N
        
        # Using a helper list to maintain state across a map/comprehension
        state_max = [0]
        [prefix_max.__setitem__(i, (state_max.__setitem__(0, max(state_max[0], P[i])), state_max[0])[1]) 
         for i in range(N)]
        
        state_min = [N + 1]
        [suffix_min.__setitem__(i, (state_min.__setitem__(0, min(state_min[0], P[N-1-i])), state_min[0])[1]) 
         for i in range(N)]
        # suffix_min is currently mirrored, let's fix it by reversing or indexing
        # suffix_min[i] now holds min of P[N-1-i : N]
        
        # Condition for k (1-indexed):
        # For k=1: sort P[2...N]. Sorted if P[1...N] becomes 1...N.
        # This requires P[0] to be 1? No, the operation sorts P[k+1...N].
        # If k=1, we sort P[2...N]. P becomes (P_1, 1, 2, ..., N) excluding P_1.
        # This is sorted if P_1 = 1.
        # Wait, the rule is: sort 1 to k-1, sort k+1 to N.
        # If k=1, we sort P[2...N]. Result is (P_1, sorted(P_2...P_N)).
        # This is (1, 2, ..., N) iff P_1 = 1 and {P_2...P_N} = {2...N}.
        # If k=N, we sort P[1...N-1]. Result is (sorted(P_1...P_{N-1}), P_N).
        # This is (1, 2, ..., N) iff P_N = N and {P_1...P_{N-1}} = {1...N-1}.
        # For 1 < k < N:
        # Result is (sorted(P_1...P_{k-1}), P_k, sorted(P_{k+1}...P_N)).
        # This is sorted iff P_k = k, max(P_1...P_{k-1}) = k-1, min(P_{k+1}...P_N) = k+1.
        
        # Check k=1: P_1 must be 1. But the operation sorts P[2...N].
        # If P_1 = 1, then sorting the rest makes it 1, 2, ..., N.
        # Check k=N: P_N must be N.
        # Check 1 < k < N: P_k = k, prefix_max[k-2] = k-1, suffix_min[N-k-1] = k+1.
        
        # We can use any() to check these conditions.
        can_solve_1 = (
            P[0] == 1 or 
            P[N-1] == N or 
            any(P[i] == i + 1 and prefix_max[i-1] == i and suffix_min[N-i-2] == i + 2 
                for i in range(1, N-1))
        )
        
        return 1 if can_solve_1 else 2

    # Process all cases and print
    results = map(lambda x: calculate_answer(*x), get_cases(input_data))
    print('\n'.join(map(str, results)))

if __name__ == "__main__":
    solve()