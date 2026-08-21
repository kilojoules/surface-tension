import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Pointer to track current position in input_data list
    ptr = 1
    
    results = []
    for _ in range(T):
        N = int(input_data[ptr])
        P = list(map(int, input_data[ptr + 1 : ptr + 1 + N]))
        ptr += 1 + N
        
        # The goal is to reach P_i = i for all i.
        # An operation with index k sorts [1, k-1] and [k+1, N].
        # This means P_k remains in its position, while everything else is sorted.
        # If we can find a k such that P_k = k, and after sorting the two sides,
        # all elements in [1, k-1] are < k and all elements in [k+1, N] are > k,
        # then 1 operation suffices.
        # Actually, the condition for 1 operation is simpler:
        # There exists k such that P_k = k AND 
        # (all elements in P[0...k-2] are < k) AND (all elements in P[k...N-1] are > k).
        # Wait, the sorting happens AFTER choosing k. 
        # If we choose k, the elements {P_1...P_{k-1}} are sorted and {P_{k+1}...P_N} are sorted.
        # For the result to be (1, 2, ..., N), we need:
        # 1. P_k = k
        # 2. The set {P_1, ..., P_{k-1}} must be exactly {1, ..., k-1}
        # 3. The set {P_{k+1}, ..., P_N} must be exactly {k+1, ..., N}
        # Condition 2 and 3 are satisfied if and only if P_k = k and 
        # max(P_1...P_{k-1}) < k (which implies they are 1...k-1).
        
        # Check if 0 operations are needed
        # We use a generator expression with all() for efficiency
        if all(P[i] == i + 1 for i in range(N)):
            results.append("0")
            continue
            
        # To check if 1 operation is enough:
        # We need a k (1-indexed) such that P[k-1] == k and 
        # max(P[0...k-2]) < k and min(P[k...N-1]) > k.
        # We can precompute prefix maximums and suffix minimums.
        
        pref_max = [0] * N
        curr_max = 0
        for i in range(N):
            curr_max = max(curr_max, P[i])
            pref_max[i] = curr_max
            
        suff_min = [0] * N
        curr_min = N + 1
        for i in range(N - 1, -1, -1):
            curr_min = min(curr_min, P[i])
            suff_min[i] = curr_min
            
        # k is 1-indexed, so index in P is k-1.
        # For k=1: only suffix [1...N-1] is sorted. We need P[0]=1 and suffix sorted.
        # Wait, the rule is: sort 1 to k-1, and k+1 to N.
        # If k=1, we sort 2...N. If P[0]=1, then 1 op works.
        # If k=N, we sort 1...N-1. If P[N-1]=N, then 1 op works.
        # If 1 < k < N, we need P[k-1]=k, max(P[0...k-2]) < k, and min(P[k...N-1]) > k.
        
        # Check k=1
        # P[0] == 1 is enough because sorting P[1...N-1] will make them 2...N.
        # Check k=N
        # P[N-1] == N is enough because sorting P[0...N-2] will make them 1...N-1.
        # Check 1 < k < N
        # P[i] == i+1 and pref_max[i-1] < i+1 and suff_min[i+1] > i+1
        
        possible_1 = (P[0] == 1 or P[N-1] == N or 
                      any(P[i] == i + 1 and pref_max[i-1] < i + 1 and 
                          (i == N-1 or suff_min[i+1] > i + 1) 
                          for i in range(1, N-1)))
        
        if possible_1:
            results.append("1")
        else:
            # It is proven that maximum 2 operations are always sufficient.
            # For example, k=1 then k=N.
            results.append("2")
            
    sys.stdout.write("\n".join(results) + "\n")

if __name__ == "__main__":
    solve()