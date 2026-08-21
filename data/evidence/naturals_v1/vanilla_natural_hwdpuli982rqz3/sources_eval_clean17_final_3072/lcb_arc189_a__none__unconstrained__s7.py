import sys

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))
    MOD = 998244353

    # The operation is: if X[l] == X[r] and X[i] != X[l] for l < i < r,
    # then X[i] becomes X[l] for l < i < r.
    # This is essentially merging a block of opposite values into the surrounding value.
    # The target state A can be reached if and only if A is obtainable from 
    # the initial state (1, 0, 1, 0, ...) via these operations.
    # Note: Initial state is X_i = i % 2. So X = [1, 0, 1, 0, ...].
    # A is reachable if A_i == X_i whenever A_i != A_{i-1} (roughly).
    # More formally, the sequence of blocks of identical values in A must be 
    # a subsequence of the sequence of blocks in X.
    # Since X is 1, 0, 1, 0..., the blocks in A must alternate and 
    # the k-th block in A must have the same value as the k-th block in X.
    
    # Let's compress A into blocks of identical consecutive elements.
    # blocks = [(value, length), ...]
    blocks = []
    if N > 0:
        curr_val = A[0]
        curr_len = 1
        for i in range(1, N):
            if A[i] == curr_val:
                curr_len += 1
            else:
                blocks.append((curr_val, curr_len))
                curr_val = A[i]
                curr_len = 1
        blocks.append((curr_val, curr_len))

    # Validation: The i-th block must have value (i % 2) if we index blocks from 0
    # and the first block is A[0]. Wait, the initial X is X_1=1, X_2=0...
    # So X_i = i % 2.
    # The first block of A must consist of values equal to X_1 (which is 1), 
    # unless the first block was merged into the second, but the operation 
    # only replaces l+1...r-1. The ends l and r remain.
    # Actually, the condition is: A is reachable iff A_i = X_i for all i 
    # such that A_i != A_{i+1} (for i < N) and A_N = X_N.
    # Simplified: The sequence of alternating values in A must be a suffix of 
    # the sequence of alternating values in X, and the first value of A must 
    # match the value of X at the index where the first block of A starts.
    
    # Let's use the property: an operation reduces the number of blocks by 2.
    # To get from X (N blocks) to A (M blocks), we need (N - M) // 2 operations.
    # Each operation consists of picking a block of length 1 and merging it.
    # This is equivalent to counting ways to parenthesize the reduction.
    # The number of ways to reduce a sequence of length k to 1 via this specific 
    # operation is the (k-1)-th Catalan number? No, it's simpler.
    # For a block of length L in A, it was formed by merging (L-1) blocks of 
    # opposite parity. This is only possible if the original X had 
    # alternating values.
    
    # Correct logic:
    # A is reachable if A_i = X_i for all i where A_i != A_{i-1}.
    # Let the blocks of A be B_1, B_2, ..., B_M.
    # Let L_j be the length of block B_j.
    # The number of ways to form block B_j is (L_j)! ? No.
    # The number of ways to reduce a sequence of length 2k+1 to 1 is k!.
    # No, the operations can overlap.
    # The actual answer is: if A is reachable, the answer is \prod (L_j ! / 2^( (L_j-1)//2 ))
    # Wait, the standard result for this specific problem is:
    # Let L_j be the length of the j-th block of identical elements in A.
    # The number of ways is \prod_{j=1}^M ( (L_j + 1) // 2 ) ! / (something)
    # Actually, for each block of length L, the number of ways to have 
    # constructed it is L // 2 factorial.
    
    # Let's re-evaluate: to get a block of length L, we need to perform (L-1)//2 
    # operations. The number of ways to do this is ((L-1)//2)! * 2^0? 
    # No. For a block of length 3 (1, 0, 1 -> 1, 1, 1), there is 1 way.
    # For length 5 (1, 0, 1, 0, 1 -> 1, 1, 1, 1, 1), there are 2 ways:
    # 1. Merge index 2, then merge index 3, 4.
    # 2. Merge index 4, then merge index 2, 3.
    # This is ( (L-1)//2 ) !
    
    # Check reachability:
    # X_i = i % 2. A_i must be X_i whenever A_i is the start of a block.
    # Let's check if A_i == (i+1) % 2 for all i such that i=0 or A[i] != A[i-1].
    # Note: A is 0-indexed in Python, so X_i is (i+1)%2.
    
    is_possible = True
    for i in range(N):
        if i == 0 or A[i] != A[i-1]:
            if A[i] != (i + 1) % 2:
                is_possible = False
                break
    
    if not is_possible:
        print(0)
        return

    # If possible, the answer is \prod ((L_j - 1) // 2)!
    # where L_j are the lengths of the blocks.
    
    # Precompute factorials
    max_l = N // 2 + 1
    fact = [1] * (max_l + 1)
    for i in range(2, max_l + 1):
        fact[i] = (fact[i-1] * i) % MOD
        
    ans = 1
    for val, length in blocks:
        ans = (ans * fact[(length - 1) // 2]) % MOD
        
    print(ans)

if __name__ == "__main__":
    solve()