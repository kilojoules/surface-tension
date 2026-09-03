import sys

def solve():
    # Read all input at once for speed
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
        
        # Convert P to integers
        P = [int(x) for x in P]
        
        # The goal is to find the minimum operations to make P_i = i.
        # An operation with index k sorts [1, k-1] and [k+1, N].
        # If we can find a k such that P_k = k, and all elements {1...k-1} 
        # are located in the prefix [1...k-1] (though unordered) and 
        # all elements {k+1...N} are in the suffix [k+1...N], then 1 op suffices.
        # Actually, the condition for 1 operation is:
        # There exists k such that P_k = k AND 
        # {P_1, ..., P_{k-1}} = {1, ..., k-1} AND 
        # {P_{k+1}, ..., P_N} = {k+1, ..., N}.
        # This is equivalent to saying P_k = k and max(P_1...P_{k-1}) = k-1.
        
        # However, the problem is simpler:
        # 0 operations: P is already sorted.
        # 1 operation: There exists k such that P_k = k and 
        #              (all elements < k are to the left of k) and 
        #              (all elements > k are to the right of k).
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
        # We need a k (1-indexed) such that P[k-1] == k and 
        # max(P[0...k-2]) == k-1 (if k > 1) and 
        # min(P[k...N-1]) == k+1 (if k < N).
        
        # Precompute prefix max and suffix min
        prefix_max = [0] * N
        suffix_min = [0] * N
        
        cur_max = 0
        for i in range(N):
            cur_max = max(cur_max, P[i])
            prefix_max[i] = cur_max
            
        cur_min = N + 1
        for i in range(N-1, -1, -1):
            cur_min = min(cur_min, P[i])
            suffix_min[i] = cur_min
            
        possible_1 = False
        for k_idx in range(N):
            # k = k_idx + 1
            # Condition: P[k_idx] == k
            if P[k_idx] == k_idx + 1:
                # Check prefix
                if k_idx == 0 or prefix_max[k_idx-1] == k_idx:
                    # Check suffix
                    if k_idx == N-1 or suffix_min[k_idx+1] == k_idx + 2:
                        possible_1 = True
                        break
        
        if possible_1:
            results.append("1")
        else:
            results.append("2")
            
    sys.stdout.write("\n".join(results) + "\n")

if __name__ == "__main__":
    solve()