import sys

def solve():
    """
    Problem Analysis:
    The operation consists of choosing k and sorting [1, k-1] and [k+1, N].
    This means the element at position k remains fixed, while everything else is sorted.
    
    If the permutation is already (1, 2, ..., N), the answer is 0.
    
    If we can find a k such that P_k = k, and after sorting [1, k-1] and [k+1, N], 
    the elements in [1, k-1] are exactly {1, ..., k-1} and [k+1, N] are {k+1, ..., N}, 
    then 1 operation is enough.
    Actually, the condition for 1 operation is simpler:
    Can we pick k such that P_k = k, and the set of elements {P_1, ..., P_{k-1}} 
    is exactly {1, ..., k-1}? 
    Wait, that's not quite right. The operation sorts the ranges. 
    If we pick k, the elements in positions 1 to k-1 are sorted, and k+1 to N are sorted.
    For the result to be (1, ..., N), we need:
    1. P_k must be k.
    2. The set {P_1, ..., P_{k-1}} must be {1, ..., k-1}.
    3. The set {P_{k+1}, ..., P_N} must be {k+1, ..., N}.
    
    Actually, if P_k = k and {P_1, ..., P_{k-1}} = {1, ..., k-1}, then automatically 
    {P_{k+1}, ..., P_N} = {k+1, ..., N}.
    
    What if no such k exists? 
    Can it always be done in 2 operations?
    The problem states it can be proved that it's possible.
    For N >= 3, we can always achieve the goal in at most 2 operations.
    For example:
    Op 1: k=1. P becomes (P_1, sort(P_2...P_N)). 
    Now the largest element N is at the end.
    Op 2: k=N. P becomes (sort(P_1...P_{N-1}), P_N).
    Wait, that's not quite right. 
    Let's reconsider:
    If we pick k=1, P_1 stays, and [2, N] are sorted.
    If we then pick k=N, P_N stays, and [1, N-1] are sorted.
    If we can move the correct elements to the boundaries, we can solve it.
    
    Actually, the condition for 1 operation is:
    Does there exist k in [1, N] such that P_k = k AND 
    max(P_1, ..., P_{k-1}) < k AND min(P_{k+1}, ..., P_N) > k?
    (With boundary conditions: max of empty set is 0, min of empty set is N+1).
    
    Wait, the sample 3: (3, 2, 1, 7, 5, 6, 4) -> N=7.
    P_1=3, P_2=2, P_3=1, P_4=7, P_5=5, P_6=6, P_7=4.
    None of P_k = k except P_2=2, P_5=5, P_6=6.
    For k=2: {P_1}={3} != {1}.
    For k=5: {P_1...P_4}={3,2,1,7} != {1,2,3,4}.
    For k=6: {P_1...P_5}={3,2,1,7,5} != {1,2,3,4,5}.
    So 1 operation is not enough. The answer is 2.
    
    Is 2 always enough?
    If we pick k=1, the sequence becomes (P_1, sorted(P_2...P_N)).
    If we then pick k=N, the sequence becomes (sorted(P_1...P_{N-1}), P_N).
    This doesn't necessarily work.
    But if we pick k=1, then P_N becomes N (since P_2...P_N are sorted).
    Then if we pick k=N, the range [1, N-1] is sorted, and P_N=N.
    The only thing is we need to make sure P_1 was not N before the first op.
    If P_1 = N, we can pick k=N first, then k=1.
    If P_1 != N and P_N != N, k=1 then k=N works.
    Basically, 2 operations are always sufficient for N >= 3.
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
        P = list(map(int, input_data[ptr+1 : ptr+1+N]))
        ptr += 1 + N
        
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
        # Condition: exists k such that P[k-1] == k and 
        # max(P[0...k-2]) < k and min(P[k...N-1]) > k
        
        # Precompute prefix max and suffix min
        prefix_max = [0] * N
        suffix_min = [N + 1] * N
        
        cur_max = 0
        for i in range(N):
            prefix_max[i] = cur_max
            cur_max = max(cur_max, P[i])
            
        cur_min = N + 1
        for i in range(N - 1, -1, -1):
            suffix_min[i] = cur_min
            cur_min = min(cur_min, P[i])
            
        possible_in_one = False
        for k_idx in range(N):
            # k = k_idx + 1
            if P[k_idx] == k_idx + 1:
                if prefix_max[k_idx] < k_idx + 1 and suffix_min[k_idx] > k_idx + 1:
                    possible_in_one = True
                    break
        
        if possible_in_one:
            results.append("1")
        else:
            results.append("2")
            
    sys.stdout.write("\n".join(results) + "\n")

if __name__ == '__main__':
    solve()