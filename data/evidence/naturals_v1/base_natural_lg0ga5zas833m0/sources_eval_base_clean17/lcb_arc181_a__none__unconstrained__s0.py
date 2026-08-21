import sys

def solve():
    """
    Problem Analysis:
    The operation consists of choosing an index k and sorting [1, k-1] and [k+1, N].
    This means the element at position k (P_k) remains fixed, while all other elements 
    are sorted.
    
    To achieve P_i = i for all i in 0 operations, the array must already be sorted.
    To achieve it in 1 operation, there must exist some k such that if we keep P_k fixed 
    and sort the rest, P_k ends up at its correct position (P_k = k) AND all other 
    elements are correctly placed relative to k. 
    Wait, the condition for 1 operation is simpler: if we pick k, P_k must be k, 
    and the set of elements {P_1, ..., P_{k-1}} must be {1, ..., k-1} and 
    {P_{k+1}, ..., P_N} must be {k+1, ..., N}. 
    Actually, the operation sorts the segments. So if we pick k, the result is 
    sorted if and only if P_k = k AND the elements in the first segment are 
    exactly the values 1 to k-1 (in any order) and the second segment are k+1 to N.
    
    However, the operation sorts the segments *ascendingly*. 
    If we pick k, P_k remains at index k. The elements in [1, k-1] are sorted, and 
    [k+1, N] are sorted. For the result to be (1, 2, ..., N), we must have:
    1. P_k = k
    2. The set {P_1, ..., P_{k-1}} == {1, ..., k-1}
    3. The set {P_{k+1}, ..., P_N} == {k+1, ..., N}
    
    If these hold, 1 operation is enough.
    
    What about 2 operations? 
    It is proven that any permutation can be solved. For N >= 3, 2 operations are 
    always sufficient. Why? 
    Pick k=1: P becomes (P_1, sorted(P_2...P_N)). 
    Pick k=N: P becomes (sorted(P_1...P_{N-1}), P_N).
    Actually, a simpler strategy: 
    Pick k=1: P_1 stays, [2, N] sorted.
    Pick k=N: [1, N-1] sorted, P_N stays.
    If we do k=1 then k=N, we sort [2, N] then sort [1, N-1]. 
    This might not work for all. But the problem says it's always possible.
    For N >= 3, the answer is always 0, 1, or 2.
    
    Wait, let's re-verify if 2 is always the maximum.
    If we pick k=1, the array becomes (P_1, sorted(P_2...P_N)).
    Then we pick k=N, the array becomes (sorted(P_1, sorted(P_2...P_{N-1})), P_N).
    This doesn't necessarily sort the whole thing.
    
    Correct logic for 1 operation:
    There exists k such that P_k = k and max(P_1...P_{k-1}) < k and min(P_{k+1}...P_N) > k.
    This is equivalent to saying the prefix [1, k-1] contains exactly the numbers 1 to k-1.
    
    Wait, let's check Sample 3: (3, 2, 1, 7, 5, 6, 4)
    k=1: P_1=3. Not 1.
    k=2: P_2=2. Prefix {3}. Not {1}.
    k=3: P_3=1. Not 3.
    k=4: P_4=7. Not 4.
    k=5: P_5=5. Prefix {3,2,1,7}. Not {1,2,3,4}.
    k=6: P_6=6. Prefix {3,2,1,7,5}. Not {1,2,3,4,5}.
    k=7: P_7=4. Not 7.
    None work. Answer is 2.
    
    Is 2 always enough?
    If we pick k=1, P becomes (P_1, sorted(P_2...P_N)).
    If we then pick k=N, the first N-1 elements are sorted.
    The only element that could be misplaced is P_N.
    Actually, the operation is very powerful. 
    If we pick k, we essentially "freeze" P_k and sort everything else.
    If we pick k=1, we sort [2, N]. If we then pick k=N, we sort [1, N-1].
    This will definitely sort the array for N >= 3.
    Example: (3, 2, 1)
    k=1: (3, 1, 2)
    k=3: (1, 2, 3) - Done.
    
    So the algorithm is:
    1. If P is already sorted, answer 0.
    2. If there exists k such that P_k = k and {P_1...P_{k-1}} = {1...k-1}, answer 1.
    3. Otherwise, answer 2.
    """
    
    # Use fast I/O
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    ptr = 1
    
    results = []
    for _ in range(T):
        N = int(input_data[ptr])
        P = input_data[ptr+1 : ptr+1+N]
        ptr += 1 + N
        
        # Convert to ints
        P = [int(x) for x in P]
        
        # Check if already sorted
        is_sorted = True
        for i in range(N):
            if P[i] != i + 1:
                is_sorted = False
                break
        
        if is_sorted:
            results.append("0")
            continue
        
        # Check if 1 operation is enough
        # We need k such that P[k-1] == k and max(P[0...k-2]) == k-1
        # and min(P[k...N-1]) == k+1.
        # Actually, if P[k-1] == k and max(P[0...k-2]) == k-1, 
        # then the first k-1 elements must be a permutation of 1...k-1.
        # Then the remaining elements must be k+1...N.
        
        found_one = False
        
        # Prefix maxes
        pref_max = [0] * N
        curr_max = 0
        for i in range(N):
            if P[i] > curr_max:
                curr_max = P[i]
            pref_max[i] = curr_max
            
        # Suffix mins
        suff_min = [0] * N
        curr_min = N + 1
        for i in range(N - 1, -1, -1):
            if P[i] < curr_min:
                curr_min = P[i]
            suff_min[i] = curr_min
            
        for k_idx in range(N):
            # k = k_idx + 1
            # P[k_idx] is the frozen element
            if P[k_idx] == k_idx + 1:
                # Check prefix
                if k_idx == 0 or pref_max[k_idx - 1] == k_idx:
                    # Check suffix
                    if k_idx == N - 1 or suff_min[k_idx + 1] == k_idx + 2:
                        found_one = True
                        break
        
        if found_one:
            results.append("1")
        else:
            results.append("2")
            
    sys.stdout.write("\n".join(results) + "\n")

if __name__ == "__main__":
    solve()