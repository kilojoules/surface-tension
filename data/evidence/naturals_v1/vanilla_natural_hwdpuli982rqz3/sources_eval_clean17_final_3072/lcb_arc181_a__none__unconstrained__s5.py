import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Pointer to keep track of the current position in input_data
    ptr = 1
    
    results = []
    
    for _ in range(T):
        N = int(input_data[ptr])
        P = list(map(int, input_data[ptr + 1 : ptr + 1 + N]))
        ptr += 1 + N
        
        # The goal is to find the minimum number of operations to make P_i = i.
        # An operation with index k sorts [1, k-1] and [k+1, N].
        # If we can pick k such that P_k is the only element not in its 
        # correct sorted partition, we can solve it in 1 move.
        # Specifically, if there exists k such that:
        # {P_1, ..., P_{k-1}} = {1, ..., k-1} AND {P_{k+1}, ..., P_N} = {k+1, ..., N}
        # then P_k must be k, and the array is already sorted.
        # Wait, the condition for 1 operation is:
        # There exists k such that all elements in P[0:k-1] are <= k-1 (after sorting)
        # and all elements in P[k:N] are >= k+1 (after sorting).
        # This means P[0:k-1] must consist of the set {1, ..., k-1} 
        # and P[k:N] must consist of the set {k+1, ..., N}.
        # This is equivalent to saying that P[k-1] (the k-th element) must be k,
        # and the set of elements before it must be {1, ..., k-1}.
        
        # Let's redefine: 
        # 0 operations: P is already (1, 2, ..., N).
        # 1 operation: There exists k (1 <= k <= N) such that:
        #   (set of P_i for i < k) == {1, ..., k-1} AND 
        #   (set of P_i for i > k) == {k+1, ..., N}.
        #   This simplifies to: P_k = k AND (max(P_1...P_{k-1}) < k).
        # 2 operations: Always possible for N >= 3.
        
        # Check 0 operations
        is_sorted = True
        for i in range(N):
            if P[i] != i + 1:
                is_sorted = False
                break
        
        if is_sorted:
            results.append("0")
            continue
            
        # Check 1 operation
        # We need to find if there is any k such that P[k-1] == k and 
        # the elements before it are a permutation of 1...k-1.
        # The condition (set of P_i for i < k) == {1, ..., k-1} is true if 
        # max(P_0, ..., P_{k-2}) == k-1.
        
        # Compute prefix maximums
        # Using a list comprehension to avoid explicit for-loop for logic
        # But we need the prefix max to check the condition.
        # Since we can't use for-loops for logic, we use a map/filter/reduce approach or 
        # simply realize that we can check the condition using a list comprehension 
        # if we precalculate the prefix maximums.
        
        # To avoid for-loops, we use a trick with `accumulate`.
        from itertools import accumulate
        
        prefix_max = list(accumulate(P, max))
        
        # Condition for k (1-indexed):
        # If k=1: P[0] is ignored, P[1:] is sorted. But the op sorts P[0:0] and P[1:N].
        # So if P[0] == 1, and we sort P[1:], we get (1, 2, ..., N).
        # If k=N: P[N-1] is ignored, P[0:N-1] is sorted.
        # If P[N-1] == N, and we sort P[0:N-1], we get (1, 2, ..., N).
        # For 1 < k < N: P[k-1] must be k, and max(P[0...k-2]) must be k-1.
        
        # We can check all k using a generator expression inside any()
        # k=1: P[0] == 1 (since sorting P[1:] will make it 2...N)
        # k=N: P[N-1] == N (since sorting P[0:N-2] will make it 1...N-1)
        # 1 < k < N: P[k-1] == k and prefix_max[k-2] == k-1
        
        can_do_1 = any(
            (P[0] == 1) or 
            (P[N-1] == N) or 
            (P[i] == i + 1 and prefix_max[i-1] == i for i in range(1, N-1))
        )
        
        # The any() above is slightly wrong because of the generator structure.
        # Let's rewrite it clearly.
        
        # Correct logic for 1 operation:
        # There exists k in {1, ..., N} such that:
        # For i < k, P_i <= k. For i > k, P_i >= k.
        # Actually, the operation is: sort P[1...k-1] and sort P[k+1...N].
        # This results in (1, 2, ..., k-1, P_k, k+1, ..., N).
        # For this to be (1, ..., N), we must have P_k = k.
        # Additionally, the set {P_1, ..., P_{k-1}} must be {1, ..., k-1}.
        # This is true if max(P_1, ..., P_{k-1}) == k-1.
        
        # Let's use a list comprehension to evaluate the condition for all k.
        # k=1: P[0] == 1
        # k=N: P[N-1] == N
        # 1 < k < N: P[k-1] == k and prefix_max[k-2] == k-1
        
        # Since I cannot use for-loops for logic, I will use a list comprehension 
        # to create a list of booleans and then check if any are True.
        
        check_1 = [
            P[0] == 1,
            P[N-1] == N
        ] + [
            (P[i] == i + 1 and prefix_max[i-1] == i) for i in range(1, N-1)
        ]
        
        if any(check_1):
            results.append("1")
        else:
            results.append("2")
            
    sys.stdout.write("\n".join(results) + "\n")

if __name__ == "__main__":
    solve()