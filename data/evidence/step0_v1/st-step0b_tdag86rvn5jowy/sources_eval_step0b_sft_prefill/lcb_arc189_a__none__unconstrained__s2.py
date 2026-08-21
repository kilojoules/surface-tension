```python
import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))
    MOD = 998244353

    # The operation is: choose l, r such that X[l] == X[r] and X[i] != X[l] for l < i < r.
    # Then X[i] becomes X[l] for l < i < r.
    # This is essentially merging a block of opposite values into the surrounding value.
    # The target state A is reachable if it can be formed by repeatedly replacing 
    # "010" with "000" or "101" with "111".
    # This is equivalent to saying that A must be obtainable by taking the initial 
    # sequence (1, 0, 1, 0, ...) and performing these contractions.
    # Notice that the operation preserves the values at the boundaries of contiguous blocks.
    # Let the initial sequence be S. S = [1, 0, 1, 0, ...].
    # The operation removes a block of length 1 (or more, if already converted) 
    # that is sandwiched between two identical values.
    
    # Let's compress A into blocks of identical consecutive elements.
    # A = [1, 1, 1, 1, 1, 0] -> blocks: (1, 5), (0, 1)
    # The initial sequence S has N blocks of size 1.
    # To reach A, we must have merged blocks.
    # Specifically, if we have a block of value V and length L in A, 
    # it must have come from a segment in S that started and ended with V.
    # The number of ways to form a block of length L using these operations 
    # is the number of ways to parenthesize the contractions.
    # This is related to Catalan numbers. Specifically, a block of length L 
    # in A corresponds to a range in S of length 2L-1 (if the block is at the end) 
    # or 2L+1 (if internal).
    
    # Correct logic:
    # The operation is: (V, !V, V) -> (V, V, V).
    # This is like removing the middle !V.
    # To get a contiguous block of length k of value V, we need to have started with 
    # a sequence like V, !V, V, !V, V ... V (k times).
    # This requires k blocks of V and k-1 blocks of !V.
    # Total length in S = k + (k-1) = 2k - 1.
    # The number of ways to reduce this to a single block of length 2k-1 is 
    # the (k-1)-th Catalan number: C_{k-1} = (1/(k)) * binom(2k-2, k-1).
    
    # However, we must check if A is reachable.
    # A is reachable if A_i matches the parity of the block index.
    # Let's represent A as a sequence of block lengths: l_1, l_2, ..., l_m.
    # The first block A[0] must match S[0] (which is 1). If A[0] == 0, 0 ways.
    # The i-th block must be "reducible" from the corresponding segment of S.
    # S = 1, 0, 1, 0, 1, 0...
    # Block 1 (value A[0]) covers S[0 ... 2*l_1 - 2]
    # Block 2 (value A[1]) covers S[2*l_1 - 1 ... 2*(l_1 + l_2) - 2]
    # And so on.
    # The total length covered is sum(2*l_i - 1) = 2*N - m.
    # But the total length is N. So 2*N - m = N => m = N.
    # This means we can only have blocks of length 1? No.
    # The operation is: replace X[l+1...r-1] with X[l].
    # This means if we have 1, 0, 1, the 0 becomes 1.
    # To get a block of length L, we need to perform L-1 operations.
    # Each operation consumes one "middle" element.
    # To get A, we can group A into contiguous blocks of identical values.
    # Let the blocks be B_1, B_2, ..., B_m with lengths L_1, L_2, ..., L_m.
    # For this to be possible:
    # 1. A[0] must be 1 (since S[0] = 1).
    # 2. The total number of elements "consumed" must be N - (sum of L_i where L_i are the 
    #    lengths of blocks that were NOT expanded).
    # Actually, the simplest way:
    # Each block i of length L_i in A is formed by taking a range in S of length 2*L_i - 1
    # and collapsing it. The number of ways to do this is C_{L_i - 1}.
    # The total length used is sum(2*L_i - 1) = 2*N - m.
    # This must equal N. So m = N. This means L_i = 1 for all i.
    # Wait, the sample 1: A = [1, 1, 1, 1, 1, 0]. Blocks: (1, 5), (0, 1).
    # L_1 = 5, L_2 = 1. Sum(2*L_i - 1) = (10-1) + (2-1) = 10. But N=6.
    # The constraint is that we can only collapse if the middle is DIFFERENT.
    # If we have 1, 0, 1, 0, 1, we can collapse the 0s.
    # To get 5 ones, we need 1, 0, 1, 0, 1. (Length 5).
    # Then the remaining is 0. Total length 6.
    # So a block of length L requires a minimum of L elements in S if they are already 
    # the correct color, but we can "absorb" opposite colors.
    # To get a block of length L of color V, we need L elements of color V, 
    # and we can absorb any number of blocks of color !V between them.
    # But the operation says: "replace each ... with X[l]".
    # This means (1, 0, 1) -> (1, 1, 1). The 0 is gone, and we get two additional 1s.
    # No, the 0 is replaced by 1. So (1, 0, 1) becomes (1, 1, 1).
    # The length does not change. The number of 1s increases by 1, 0s decrease by 1.
    # To get a block of length L, we need to have performed (L - count of V in that range) operations.
    # Each operation requires a "sandwich" (V, !V, V).
    
    # Correct combinatorial approach:
    # A block of length L of color V is formed from a sequence of alternating colors.
    # The number of ways to form a block of length L is the (L-1)-th Catalan number C_{L-1}.
    # This is because each operation reduces the number of blocks by 2.
    # To get a single block of length L, we must have started with L blocks of color V 
    # and L-1 blocks of color !V.
    # Total length = L + (L-1) = 2L - 1.
    # The total length N must be sum(2*L_i - 1) for i=1...m, but the last block 
    # might be truncated.
    # Let L_1, ..., L_m be the lengths of contiguous blocks in A.
    # The first m-1 blocks must have been formed by (2*L_i - 1) elements of S.
    # The last block L_m is formed by the remaining elements.
    # For the last block to be valid, the remaining length must be at least L_m 
    # and have the same parity as L_m (since it's alternating).
    # Actually, the last block L_m is formed by the remaining N - sum_{i=1}^{m-1}(2*L_i - 1) elements.
    # Let this remaining length be R. We need to find the number of ways to turn 
    # an alternating sequence of length R into a block of length L_m.
    # This is possible if R = 2*L_m - 1 or R = 2*L_m.
    # If R = 2*L_m - 1, ways = C_{L_m - 1}.
    # If R = 2*L_m, ways = C_{L_m - 1} (the extra element must be the opposite color).
    
    # Let's refine:
    # 1. Group A into blocks of length L_1, ..., L_m.
    # 2. Check if A[0] == 1. If not, 0.
    # 3. The first L_1 elements of A come from the first 2*L_1 - 1 elements of S.
    # 4. The second L_2 elements of A come from the next 2*L_2 - 1 elements of S...
    # 5. The last block L_m comes from the remaining R = N - sum_{i=1}^{m-1} (2*L_i - 1) elements.
    # 6. For each i < m, we need 2*L_i - 1 <= R_current.
    # 7. For the last block, we need R >= L_m and (R - L_m) must be even (to be absorbed).
    #    Wait, if R = 2*L_m - 1, we have L_m ones and L_m-1 zeros.
    #    If R = 2*L_m, we have L_m ones and L_m zeros. The last zero cannot be absorbed.
    #    But the operation is: choose l, r. The last element can't be l.
    #    If the sequence is 1, 0, 1, 0, we can make it 1, 1, 1, 0.
    #    The last element remains.
    #    So if R = 2*L_m, the last element must be different from A[m-1].
    #    If R = 2*L_m - 1, the last element must be A[m-1].
    
    # Let's calculate:
    # For i = 1 to m-1: ways *= C_{L_i - 1}
    # For i = m: 
    #   R = N - sum_{j=1}^{m-1} (2*L_j - 1)
    #   If R < L_m or (R - L_m) is odd: return 0
    #   If R == 2*L_m - 1: ways *= C_{L_m - 1}
    #   If R == 2*L_m: ways *= C_{L_m - 1} (the last element is just left over)
    #   If R > 2*L_m: this is impossible because we can't "delete" elements, only overwrite.
    #   Actually, if R > 2*L_m, we can't reach the state because we'd have 
    #   more than L_m elements of the correct color or we can't overwrite the extras.
    #   Wait, if R = 2*L_m + 1, we have L_m + 1 elements of color A[m-1].
    #   That would mean the block length is L_m + 1, not L_m.
    #   So R must be exactly 2*L_m - 1 or 2*L_m.
    
    if N == 0:
        print(0)
        return

    # Group A into blocks
    blocks = []
    if N > 0:
        current_val = A[0]
        current_len = 1
        for i in range(1, N):
            if A[i] == current_val:
                current_len += 1
            else:
                blocks.append(current_len)
                current_val = A[i]
                current_len = 1
        blocks.append(current_len)

    if A[0] == 