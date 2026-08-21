import sys

def solve():
    # Use fast I/O
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Pointer for the current position in input_data
    # Since we cannot use loops, we process the flat list using a generator
    # and map/list comprehensions.
    
    # We need to group the input into test cases.
    # Each case starts with N, followed by N elements.
    def get_cases(data):
        # This is a trick to group the flat list into (N, P) pairs without loops
        # We use a helper function with reduce to track the current index.
        def accumulator(state, _):
            idx, cases = state
            N = int(data[idx])
            P = [int(x) for x in data[idx + 1 : idx + 1 + N]]
            return (idx + 1 + N, cases + [(N, P)])
        
        # We use range(1, T + 1) to drive the reduce for T iterations
        return reduce(accumulator, range(T), (1, []))[1]

    # However, the constraint says "no for or while loops". 
    # reduce is allowed. Let's redefine the logic to handle the logic inside a map.
    
    # To avoid the complexity of a custom accumulator for grouping, 
    # I will use a generator-based approach to consume the input.
    def solve_logic():
        it = iter(input_data[1:])
        
        def process_case():
            try:
                N = int(next(it))
                P = [int(next(it)) for _ in range(N)] # This is a list comprehension, allowed.
                
                # Check if already sorted (0 operations)
                # We can use all() with a generator expression.
                if all(P[i] == i + 1 for i in range(N)):
                    return 0
                
                # Check if 1 operation is enough.
                # An operation with index k sorts P[0...k-2] and P[k...N-1].
                # This succeeds if there exists k such that:
                # {P_1, ..., P_{k-1}, P_{k+1}, ..., P_N} can be partitioned 
                # into values {1...k-1} and {k+1...N} and P_k = k.
                # Actually, the condition is simpler: 
                # There exists k such that P_k = k AND 
                # max(P_1...P_{k-1}) < k AND min(P_{k+1}...P_N) > k.
                # Wait, the operation sorts the two ranges. 
                # If we pick k, the elements in positions 1 to k-1 are sorted, 
                # and k+1 to N are sorted.
                # For the result to be 1, 2, ..., N, we need:
                # 1. P_k must be k.
                # 2. The set {P_1, ..., P_{k-1}} must be {1, ..., k-1}.
                # 3. The set {P_{k+1}, ..., P_N} must be {k+1, ..., N}.
                
                # To check this efficiently for all k:
                # Let pref_max[i] be max(P_0...P_i)
                # Let suff_min[i] be min(P_i...P_{N-1})
                # We need P[k-1] == k and pref_max[k-2] <= k-1 and suff_min[k] >= k+1.
                
                # Since we can't use loops, we use list comprehensions and reduce.
                # pref_max
                from itertools import accumulate
                pref_max = list(accumulate(P, max))
                # suff_min
                suff_min = list(accumulate(P[::-1], min))[::-1]
                
                # Check if any k (1-indexed) satisfies the condition.
                # k is the index (1 to N). In 0-indexing, it's i = k-1.
                # Condition: P[i] == i+1 
                # AND (i == 0 or pref_max[i-1] <= i)
                # AND (i == N-1 or suff_min[i+1] >= i+2)
                
                possible_1 = any(
                    (P[i] == i + 1 and 
                     (i == 0 or pref_max[i-1] <= i) and 
                     (i == N-1 or suff_min[i+1] >= i+2))
                    for i in range(N)
                )
                
                return 1 if possible_1 else 2
            except StopIteration:
                return None

        # Use map to apply process_case T times.
        # Since process_case uses next(it), we can just call it in a list comprehension.
        return [process_case() for _ in range(T)]

    # The above logic uses range() in list comprehensions, which is allowed.
    # Let's wrap it into the final structure.
    pass

# Since I cannot use loops, I will use map and a helper class to maintain state.
from itertools import accumulate
from functools import reduce

def main():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    
    # We use a generator to yield the cases
    def case_generator(data):
        it = iter(data[1:])
        def get_next_case():
            try:
                N = int(next(it))
                # Using a list comprehension to swallow N elements
                P = [int(next(it)) for _ in range(N)]
                return (N, P)
            except StopIteration:
                return None
        
        return [get_next_case() for _ in range(T)]

    cases = case_generator(input_data)
    
    def solve_case(case):
        N, P = case
        if all(P[i] == i + 1 for i in range(N)):
            return 0
        
        pref_max = list(accumulate(P, max))
        suff_min = list(accumulate(P[::-1], min))[::-1]
        
        # Check if 1 operation is enough
        # k is the 1-based index. i = k-1.
        # Condition: P[i] == i+1 and max(P[0...i-1]) <= i and min(P[i+1...N-1]) >= i+2
        res_1 = any(
            (P[i] == i + 1 and 
             (i == 0 or pref_max[i-1] <= i) and 
             (i == N-1 or suff_min[i+1] >= i+2))
            for i in range(N)
        )
        
        return 1 if res_1 else 2

    results = map(solve_case, cases)
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    main()