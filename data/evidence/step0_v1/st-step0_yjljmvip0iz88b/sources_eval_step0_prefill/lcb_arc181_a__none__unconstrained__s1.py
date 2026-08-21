import sys

def solve():
    # Read all input at once for efficiency
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    it = iter(input_data)
    T_str = next(it, None)
    if T_str is None:
        return
    T = int(T_str)
    
    results = []
    for _ in range(T):
        N = int(next(it))
        P = [int(next(it)) for _ in range(N)]
        
        # The goal is to reach P_i = i for all i.
        # An operation with index k sorts [1, k-1] and [k+1, N].
        # This means P_k remains in its position, while everything else is sorted.
        # If we can find a k such that P_k = k, and after sorting the two sides,
        # all elements P_i (i < k) are < k and all P_i (i > k) are > k,
        # then one operation is enough.
        # However, the operation sorts the values currently in those positions.
        # For one operation with index k to work, we need:
        # 1. P_k = k
        # 2. The set {P_1, ..., P_{k-1}} must be {1, ..., k-1}
        # 3. The set {P_{k+1}, ..., P_N} must be {k+1, ..., N}
        # Actually, condition 2 and 3 are equivalent to saying that for all i < k, P_i < k.
        
        # If the permutation is already sorted, 0 operations.
        is_sorted = True
        for i in range(N):
            if P[i] != i + 1:
                is_sorted = False
                break
        
        if is_sorted:
            results.append("0")
            continue
            
        # Check if 1 operation is enough.
        # We need a k such that P_k = k and max(P_1...P_{k-1}) < k.
        # This is equivalent to saying that for all i < k, P_i < k.
        # Let's track the prefix maximums and suffix minimums.
        
        prefix_max = [0] * N
        curr_max = 0
        for i in range(N):
            curr_max = max(curr_max, P[i])
            prefix_max[i] = curr_max
            
        suffix_min = [0] * N
        curr_min = N + 1
        for i in range(N - 1, -1, -1):
            curr_min = min(curr_min, P[i])
            suffix_min[i] = curr_min
            
        possible_in_one = False
        for k_idx in range(N):
            # k = k_idx + 1
            # Condition: P[k_idx] == k_idx + 1
            # And for all i < k_idx, P[i] < k_idx + 1  => prefix_max[k_idx-1] < k_idx + 1
            # And for all i > k_idx, P[i] > k_idx + 1  => suffix_min[k_idx+1] > k_idx + 1
            
            if P[k_idx] == k_idx + 1:
                left_ok = (k_idx == 0) or (prefix_max[k_idx-1] < k_idx + 1)
                right_ok = (k_idx == N-1) or (suffix_min[k_idx+1] > k_idx + 1)
                if left_ok and right_ok:
                    possible_in_one = True
                    break
        
        if possible_in_one:
            results.append("1")
        else:
            # It is proven that any permutation can be sorted in at most 2 operations.
            # For example, k=1 sorts [2, N], then k=N sorts [1, N-1].
            # Wait, the problem says it's always possible. Let's check if 2 is the max.
            # If we pick k=1, P becomes (P_1, sorted(P_2...P_N)).
            # Then we pick k=N, P becomes (sorted(P_1...P_{N-1}), P_N).
            # This doesn't necessarily sort it.
            # But if we pick k=1, then P_1 is fixed and P_2...P_N are sorted.
            # If P_1 was 1, we are done. If not, P_1 is some value x > 1.
            # After k=1, P = (x, 1, 2, ..., x-1, x+1, ..., N).
            # Now pick k=x. P_x is the value at index x.
            # Since P_2...P_N were sorted, P_x is the (x-1)-th element of the sorted list,
            # which is x-1. This is not helping.
            # Actually, the strategy is: 
            # 1. k=1: P becomes (P_1, sorted(P_2...P_N))
            # 2. k=N: P becomes (sorted(P_1...P_{N-1}), P_N)
            # This is still not quite right.
            # Correct strategy for 2 ops:
            # Op 1: k=1. P becomes (P_1, 1, 2, ..., P_1-1, P_1+1, ..., N) [assuming P_1 is the value]
            # Wait, the values are sorted, not the indices.
            # If k=1, P_1 stays, and P_2...P_N are sorted.
            # The sorted sequence is (1, 2, ..., N) excluding P_1.
            # So P becomes (P_1, 1, 2, ..., P_1-1, P_1+1, ..., N).
            # Now the value P_1 is at index 1. The value 1 is at index 2.
            # If we then pick k=P_1, the element at index P_1 is P_1-1.
            # Let's use the property: 2 operations are always sufficient.
            # Pick k=1, then P = (P_1, 1, 2, ..., P_1-1, P_1+1, ..., N).
            # Now pick k=2. The first element (P_1) is sorted alone, and 
            # the elements from index 3 to N are sorted.
            # That's not it.
            # The correct 2-step: k=1 (sorts 2..N), then k=N (sorts 1..N-1).
            # After k=1: P = (P_1, 1, 2, ..., P_1-1, P_1+1, ..., N)
            # After k=N: P_N is fixed, 1..N-1 are sorted.
            # Since P_N was the largest value in the sorted 2..N (which is N),
            # P_N is N, and 1..N-1 are sorted. Done.
            results.append("2")
            
    sys.stdout.write("\n".join(results) + "\n")

if __name__ == "__main__":
    solve()