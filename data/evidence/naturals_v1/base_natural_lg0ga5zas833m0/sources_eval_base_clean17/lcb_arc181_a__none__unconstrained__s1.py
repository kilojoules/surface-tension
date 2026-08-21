import sys

def solve():
    """
    Problem Analysis:
    The operation involves choosing an index k and sorting P[1...k-1] and P[k+1...N].
    
    - If P is already (1, 2, ..., N), answer is 0.
    - If there exists k such that sorting P[1...k-1] and P[k+1...N] results in (1, 2, ..., N),
      the answer is 1.
      This happens if P[k] is already in its correct position (P[k] == k) AND 
      the set of elements {P_1, ..., P_{k-1}} is exactly {1, ..., k-1} 
      (which implies the remaining elements are {k+1, ..., N}).
      Wait, that's not quite right. The operation sorts the segments. 
      If we pick k, then P[1...k-1] becomes sorted and P[k+1...N] becomes sorted.
      The final result is (1, 2, ..., N) if and only if:
      1. P[k] == k
      2. The set {P_1, ..., P_{k-1}} contains exactly the values {1, ..., k-1}.
      3. The set {P_{k+1}, ..., P_N} contains exactly the values {k+1, ..., N}.
      Actually, if P[k] == k and condition 2 is true, condition 3 is automatically true.
      
    - If no such k exists, can we always do it in 2?
      The problem states it's always possible. Let's check if 2 is always sufficient.
      If we pick k=1, we sort P[2...N]. Then we pick k=N, we sort P[1...N-1].
      But we need to be careful.
      Actually, if we pick k=1, P becomes (P_1, sorted(P_2...P_N)).
      If we then pick k=N, P becomes (sorted(P_1...P_{N-1}), P_N).
      This doesn't necessarily result in (1...N).
      
      Correct logic for 1 operation:
      There exists k such that P[k] == k and max(P[1...k-1]) < k and min(P[k+1...N]) > k.
      Since it's a permutation, max(P[1...k-1]) < k is equivalent to saying {P_1...P_{k-1}} = {1...k-1}.
      
      Correct logic for 2 operations:
      Is it always 2 if not 0 or 1?
      Consider the constraints. N >= 3.
      If we pick k=1, P becomes (P_1, sorted(P_2...P_N)).
      If we then pick k=2, we sort P[1...1] (already sorted) and P[3...N].
      This is not helping.
      
      Let's re-evaluate:
      Operation k: P[1...k-1] sorted, P[k+1...N] sorted.
      If we pick k=1, P becomes (P_1, 2, 3, ..., N) if P_1 was 1.
      If we pick k=1, then k=N, then k=1...
      Wait, if we pick k=1, P[2...N] is sorted. If we then pick k=N, P[1...N-1] is sorted.
      If we pick k=1, then k=N, the sequence becomes:
      (P_1, P_2, ..., P_N) --k=1--> (P_1, sort(P_2...P_N))
      Then --k=N--> (sort(P_1, sort(P_2...P_{N-1})), P_N')
      
      Actually, a simpler observation:
      If we can't do it in 1, can we do it in 2?
      Pick k=1: P becomes (P_1, sort(P_2...P_N)).
      Now the element 1 is either at index 1 or index 2.
      If we then pick k=N, the first N-1 elements are sorted.
      If 1 was at index 1 or 2, it will move to index 1.
      This is still confusing.
      
      Let's use the property:
      The answer is 0 if P is already sorted.
      The answer is 1 if there is some k such that P[k] == k and {P_1...P_{k-1}} = {1...k-1}.
      Otherwise, the answer is 2.
      Why 2? Because we can pick k=1, then P becomes (P_1, sort(P_2...P_N)).
      Then pick k=N, P becomes (sort(P_1, sorted_part), P_N).
      Wait, if we pick k=1, then P[2...N] is sorted.
      Then we pick k=2, P[1...1] is sorted and P[3...N] is sorted.
      The only way this fails is if we can't move the elements.
      But for N >= 3, we can always achieve it in 2.
      Example 3: (3, 2, 1, 7, 5, 6, 4).
      k=4: (3, 2, 1) sorted -> (1, 2, 3), (5, 6, 4) sorted -> (4, 5, 6). Result: (1, 2, 3, 7, 4, 5, 6).
      Then k=3: (1, 2) sorted -> (1, 2), (7, 4, 5, 6) sorted -> (4, 5, 6, 7). Result: (1, 2, 3, 4, 5, 6, 7).
      Answer 2.
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
        ptr += N + 1
        
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
        # (Using 0-indexing, so P[k-1] is the k-th element)
        possible_in_one = False
        
        # Prefix max
        prefix_max = [0] * N
        curr_max = 0
        for i in range(N):
            curr_max = max(curr_max, P[i])
            prefix_max[i] = curr_max
            
        # Suffix min
        suffix_min = [0] * N
        curr_min = N + 1
        for i in range(N-1, -1, -1):
            curr_min = min(curr_min, P[i])
            suffix_min[i] = curr_min
            
        for k_idx in range(N):
            # The k-th element is P[k_idx]. It must be k_idx + 1.
            if P[k_idx] == k_idx + 1:
                # Elements before it must be 1...k_idx
                # Elements after it must be k_idx+2...N
                # Check prefix max for indices 0 to k_idx-1
                if k_idx == 0:
                    # k=1, only need to check suffix_min of P[1...N-1]
                    if N == 1 or suffix_min[1] == 2:
                        possible_in_one = True
                        break
                elif k_idx == N - 1:
                    # k=N, only need to check prefix_max of P[0...N-2]
                    if prefix_max[N-2] == N - 1:
                        possible_in_one = True
                        break
                else:
                    # 1 < k < N
                    if prefix_max[k_idx-1] == k_idx and suffix_min[k_idx+1] == k_idx + 2:
                        possible_in_one = True
                        break
        
        if possible_in_one:
            results.append("1")
        else:
            results.append("2")
            
    sys.stdout.write("\n".join(results) + "\n")

if __name__ == "__main__":
    solve()