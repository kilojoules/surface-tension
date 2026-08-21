```python
import sys
from itertools import groupby

def solve():
    # Read N and A
    input_data = sysstdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    a = list(map(int, input_data[1:]))
    
    # Initial state X_i = i % 2 (1-indexed)
    # X = [1, 0, 1, 0, ...] if N=6 -> [1, 0, 1, 0, 1, 0]
    # The operation: if X[l] == X[r] and X[i] != X[l] for l < i < r,
    # then X[l+1...r-1] become X[l].
    # This is essentially a range-fill operation.
    # Crucially, the condition "X[i] different from X[l]" means we can only 
    # overwrite a block of the opposite value.
    # This structure is equivalent to removing "peaks" or "valleys" in a 
    # sequence of blocks.
    
    # Let's compress A into blocks of identical consecutive values.
    # A = [1, 1, 1, 1, 1, 0] -> [(1, 5), (0, 1)]
    blocks = [(k, sum(1 for _ in g)) for k, g in groupby(a)]
    m = len(blocks)
    
    # The initial sequence is 1, 0, 1, 0, ...
    # Any sequence of operations that results in A is valid if and only if:
    # 1. A is reachable from X.
    # 2. The operations follow the rule.
    # The rule implies we can merge blocks of the same value if they are 
    # separated by a single block of the opposite value.
    # This is like a stack-based reduction.
    # The number of ways to reduce a sequence of blocks to the target 
    # is related to Catalan-like structures or binary trees.
    # Specifically, if we have a sequence of blocks, and we merge 
    # block i and i+2 because they have the same value and block i+1 
    # is the opposite, this is like a contraction.
    
    # For a target sequence of blocks of length m, the number of ways 
    # to have reached this state is the product of Catalan(k) 
    # where k is the number of blocks "absorbed" into each final block.
    # However, the problem constraints and the operation definition 
    # suggest that we are looking for the number of ways to build the 
    # final blocks using the allowed operation.
    
    # Let's analyze the structure: we can only merge blocks of the same 
    # value if the middle is different. This is exactly the process of 
    # reducing a string by removing "ABA" -> "AAA".
    # The number of ways to reduce a sequence of length L to length M 
    # via these operations is given by the product of C((L_i - M_i)/2) 
    # where L_i is the original length and M_i is the final.
    # But the initial sequence is alternating 1, 0, 1, 0...
    # So the initial number of blocks is N.
    # Each operation reduces the number of blocks by 2.
    # Total operations = (N - m) / 2.
    # If (N - m) is odd or N < m, it's impossible.
    
    # The number of ways to perform these reductions is the product of 
    # Catalan numbers for each "mountain" of reductions.
    # A simpler combinatorial result for this specific problem:
    # The answer is the product of Catalan( (len(block) - 1) // 2 ) 
    # for blocks that were expanded, but that's for a different problem.
    
    # Correct logic for this specific operation:
    # We can only merge blocks of the same value. 
    # The total number of ways is the product of Catalan((block_len - 1) // 2)
    # for each block in the final sequence A, provided the parity 
    # and boundaries match the initial X.
    
    # Check if A is reachable:
    # Initial X: X_i = i % 2.
    # A_i must be consistent with the possibility of range fills.
    # The only way A_i is unreachable is if the boundaries are wrong.
    # But the operation allows l and r to be any indices.
    # Actually, the only constraint is that we can't change X_1 and X_N.
    # Wait, the operation says l and r are chosen, and l+1...r-1 are changed.
    # So X_1 and X_N can never be changed.
    # Initial X_1 = 1 % 2 = 1. Initial X_N = N % 2.
    # If A_1 != 1 or A_N != (N % 2), answer is 0.
    
    if a[0] != 1 or a[-1] != (n % 2):
        print(0)
        return

    # The number of ways to form a block of length 'len' from 
    # alternating bits is Catalan((len - 1) // 2).
    # This is because each operation reduces the number of blocks by 2.
    # To get a block of length k, we need (k-1)//2 operations.
    
    MOD = 998244353
    
    # Precompute factorials for Catalan
    MAX = n + 1
    fact = [1] * (2 * MAX)
    inv = [1] * (2 * MAX)
    for i in range(1, 2 * MAX):
        fact[i] = (fact[i-1] * i) % MOD
    
    inv[2 * MAX - 1] = pow(fact[2 * MAX - 1], MOD - 2, MOD)
    for i in range(2 * MAX - 2, -1, -1):
        inv[i] = (inv[i+1] * (i + 1)) % MOD
        
    def nCr(n, r):
        if r < 0 or r > n: return 0
        num = fact[n]
        den = (inv[r] * inv[n-r]) % MOD
        return (num * den) % MOD

    def catalan(k):
        return nCr(2*k, k) * pow(k+1, MOD-2, MOD) % MOD

    # For each block of length L, it must have been formed by 
    # (L-1)//2 operations. This is only possible if L is odd.
    # If any block length is even, it's impossible? 
    # No, because the blocks in A are result of merges.
    # Let's re-evaluate: the blocks in A are formed by merging.
    # A block of length L in A corresponds to (L-1)//2 merges 
    # if we started with alternating bits.
    # This requires L to be odd.
    
    # Wait, Sample 1: N=6, A=[1,1,1,1,1,0]. 
    # Blocks: (1, 5), (0, 1). 
    # L1=5, L2=1. Catalan((5-1)//2) = Catalan(2) = 2.
    # Catalan((1-1)//2) = Catalan(0) = 1.
    # Product = 2 * 1 = 2. But sample output is 3.
    # Let's re-read: "Choose cells l and r (l+1 < r)".
    # Sample 1: X=(1,0,1,0,1,0). 
    # Op 1: l=2, r=4. X becomes (1, 0, 0, 0, 1, 0).
    # Op 2: l=1, r=5. X becomes (1, 1, 1, 1, 1, 0).
    # This means the blocks can be merged in different orders.
    # This is exactly the number of binary trees with k internal nodes,
    # but the "nodes" here are the operations.
    # For a block of length L, the number of ways to form it is Catalan((L-1)//2).
    # But the sample says 3. Catalan(2) is 2. Where does 3 come from?
    # Ah, the operations can overlap. 
    # The number of ways to reduce a sequence of length L to 1 
    # is the (L-1)-th Schroder number? No.
    # Let's use the formula: the number of ways is nCr(L-1, (L-1)//2) / ((L-1)//2 + 1)
    # No, that's Catalan. 
    # Let's look at the sample again. L=5. Catalan(2)=2. 
    # Maybe the answer is nCr(L-1, (L-1)//2)? nCr(4, 2) = 6. 
    # Maybe it's the number of ways to parenthesize?
    # Actually, the number of ways to reduce a sequence of length L 
    # to 1 using this operation is the Catalan number C_{(L-1)/2}.
    # But the sample says 3. Let me re-calculate Catalan(2).
    # C_0 = 1, C_1 = 1, C_2 = 2, C_3 = 5.
    # Wait, the sample output is 3. 3 is not a Catalan number.
    # Is it (L+1)//2 ? (5+1)//2 = 3.
    # Let's check Sample 2: N=10, A=[1,1,1,1,1,0,1,1,1,0]
    # Blocks: (1, 5), (0, 1), (1, 3), (0, 1)
    # Lengths: 5, 1, 3, 1.
    # If the answer is product of (L+1)//2: 3 * 1 * 2 * 1 = 6.
    # Sample 2 output is 9. 
    # 9 = 3 * 3. Maybe the blocks are (1, 5) and (1, 3) and (0, 1) and (0, 1)?
    # No, the blocks are 11111, 0, 111, 0.
    # Maybe the formula is nCr(L, L//2)? 
    # For L=5: nCr(5, 2) = 10. For L=3: nCr(3, 1) = 3.
    # For L=1: nCr(1, 0) = 1.
    # 10 * 1 * 3 * 1 = 30. Still not 9.
    # What about nCr(L-1, (L-1)//2) ?
    # For L=5: nCr(4, 2) = 6. For L=3: nCr(2, 1) = 2.
    # 6 * 1 * 2 * 1 = 12.
    # Wait, the only way to get 3 and 9 is if the formula for L=5 is 3 
    # and for L=3 is 3.
    # For L=5, (L+1)//2 = 3. For L=3, (L+1)//2 = 2. 3*2=6.
    # What if the formula is nCr(L, (L-1)//2) / something?
    # Let's try: for L=5, result is 3. For L=3, result is 3.
    # nCr(3, 1) = 3. For L=5, nCr(5, 2) = 10.
    # Wait! The number of ways to reduce a sequence of length L 
    # to 1 is nCr(L-1, (L-1)//2) / 2? No.
    # Let's re-read: "l+1 < r". This means the distance is at least 2.
    # This is exactly the condition for the "Mountain" problem.
    # The number of ways is nCr(L-1, (L-1)//2) / (something).
    # Actually, the number of ways to reduce a block of length L 
    # is nCr(L-1, (L-1)//2) / ( (L-1)//2 + 1 ) ... no.
    # Let's try nCr(L-1, (L-1)//2) for L=3 is 2, for L=5 is 6.
    # If the answer is 3 and 9, then for L=5 it's 3 and for L=