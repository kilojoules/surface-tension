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
    # we create a list of test case data.
    test_cases = [
        (int(input_data[ptr + i*N + 1]), 
         list(map(int, input_data[ptr + i*N + 2 : ptr + (i+1)*N + 2])))
        # This structure is tricky because N varies per test case.
        # Let's redefine how we extract test cases.
    ]
    # Wait, the above list comprehension assumes constant N. 
    # Let's use a different approach to group the input.
    pass

# Since I cannot use loops, I will use a generator to consume the input 
# and a map/list comprehension to process the T cases.

def process_all():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    
    # To handle variable N without loops, we can use a helper function 
    # with a mutable state (like a list) to track the current pointer.
    state = {"ptr": 1}
    
    def get_case():
        N = int(input_data[state["ptr"]])
        P = list(map(int, input_data[state["ptr"] + 1 : state["ptr"] + 1 + N]))
        state["ptr"] += 1 + N
        return N, P

    # We use map(get_case, range(T)) to extract all cases.
    # For each case (N, P), we calculate the answer.
    # The answer is 0 if sorted, 1 if there exists k such that 
    # sorting [1, k-1] and [k+1, N] results in [1, ..., N].
    # This happens if P[k-1] is the only element not in its sorted 
    # relative position, or more simply:
    # If we pick k, the elements {P_1...P_{k-1}} must be {1...k-1} 
    # and {P_{k+1}...P_N} must be {k+1...N}.
    # This is equivalent to saying:
    # max(P_1...P_{k-1}) == k-1 AND min(P_{k+1}...P_N) == k+1.
    
    def calculate_ans(case):
        N, P = case
        if P == sorted(P):
            return 0
        
        # Prefixes: max of first i elements
        # Suffixes: min of last i elements
        # We use accumulate to avoid loops.
        pref_max = list(accumulate(P, max))
        suff_min = list(accumulate(P[::-1], min))[::-1]
        
        # For a given k (1-indexed):
        # If k=1: we only need min(P[1:]) == 2
        # If k=N: we only need max(P[:N-1]) == N-1
        # If 1 < k < N: we need max(P[:k-1]) == k-1 and min(P[k:]) == k+1
        
        # Check k=1
        cond1 = (suff_min[1] == 2)
        # Check k=N
        condN = (pref_max[N-2] == N-1)
        # Check 1 < k < N
        # For k in 2...N-1, index i = k-1 (0-indexed)
        # We need pref_max[i-1] == i and suff_min[i+1] == i+2
        # Note: P is 0-indexed, so P[i] is the k-th element.
        # The condition is: max(P[0...i-1]) == i and min(P[i+1...N-1]) == i+2
        cond_mid = any(
            pref_max[i-1] == i and suff_min[i+1] == i+2 
            for i in range(1, N-1)
        )
        
        # The problem says we can use the operation multiple times.
        # However, the operation is very powerful. 
        # If we can't do it in 1, we can always do it in 2.
        # Why? Pick k=1 (sorts P[2...N]), then pick k=N (sorts P[1...N-1]).
        # Actually, the only way it takes 0 is if already sorted.
        # It takes 1 if there exists k such that the remaining parts are the correct sets.
        # Otherwise, it takes 2.
        
        return 1 if (cond1 or condN or cond_mid) else 2

    # Use map to apply calculate_ans to all cases and join with newlines
    results = map(calculate_ans, map(get_case, range(T)))
    sys.stdout.write("\n".join(map(str, results)) + "\n")

if __name__ == "__main__":
    process_all()