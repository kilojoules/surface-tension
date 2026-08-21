```python
import sys
from functools import reduce

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    k = int(input_data[1])

    # The total number of good sequences is S = (NK)! / (K!)^N
    # We need the floor((S+1)/2)-th sequence.
    # This is exactly the middle sequence (or the one just before the middle if S is even).
    # Because the set of good sequences is symmetric (replacing x with N-x+1),
    # the "middle" sequence is the one that is lexicographically "central".
    # Specifically, if a sequence is A, its complement is A' where A'_i = N + 1 - A_i.
    # A < A' if and only if A is in the first half of the lexicographical order.
    # The floor((S+1)/2)-th sequence is the largest sequence A such that A <= A'.
    # This means for the first index i where A_i != A'_i, we must have A_i < A'_i.
    # To maximize A while keeping A <= A', we want A_i to be as large as possible
    # but still satisfy the condition.
    
    # The condition A <= A' is satisfied if at the first index i where A_i != A'_i,
    # A_i < A'_i. To find the largest such A, we want A_i to be as large as possible.
    # The "middle" of the lexicographical range is reached when we try to keep
    # the sequence as balanced as possible around the value (N+1)/2.
    
    # For a fixed N and K, the floor((S+1)/2)-th sequence is constructed by:
    # For each position, try the largest possible digit d (from N down to 1).
    # If we place d, we must check if the number of sequences starting with the 
    # current prefix is <= S/2.
    # However, there is a much simpler combinatorial property:
    # The sequence is the one that "balances" the digits.
    # For N=2, K=2: (1,1,2,2), (1,2,1,2), (1,2,2,1) | (2,1,1,2), (2,1,2,1), (2,2,1,1)
    # The 3rd is (1,2,2,1).
    
    # The general rule for the floor((S+1)/2)-th sequence:
    # It is the sequence that is "lexicographically largest" among those A where A <= A'.
    # This is achieved by filling the sequence such that we use digits in a way that
    # we stay just below the "halfway" point.
    # Specifically, the sequence is:
    # For i from 1 to N:
    #   If i < (N+1)/2: use i, K times.
    #   If i > (N+1)/2: use i, K times.
    #   If i == (N+1)/2: this is the pivot.
    # Actually, the pattern is: 
    # All digits < (N+1)/2 are placed as late as possible.
    # All digits > (N+1)/2 are placed as early as possible.
    # The digit (N+1)/2 (if it exists) is placed in the middle.
    
    # Correct logic for floor((S+1)/2)-th:
    # It is the sequence where we use digits 1...N.
    # The sequence is: 
    # 1. Digits from ceil(N/2) + 1 to N, each K times, in increasing order? No.
    # Let's use the property: the sequence is the one that is "halfway".
    # For N=3, K=3: 2 2 2 1 3 3 3 1 1
    # Pattern: 
    # Mid value M = (N+1)//2.
    # Sequence: (M)*K, (1)*K, (2)*K ... (M-1)*K, (M+1)*K ... (N)*K, (M)*K ... 
    # Wait, the Sample 4 (N=3, K=3) is 2 2 2 1 3 3 3 1 1.
    # That is: M=2. Sequence: 2(K), 1(K), 3(K), 1(K) -- no, that's not it.
    # Sample 4: 2 2 2 1 3 3 3 1 1. 
    # Digits: 2 appears 3 times, 1 appears 2 times, 3 appears 3 times, 1 appears 1 time.
    # Total: 2s: 3, 1s: 3, 3s: 3. Correct.
    # The pattern is: 
    # M = (N+1)//2
    # Result: [M]*K + [1]*K + [2]*K ... [M-1]*K + [M+1]*K ... [N]*K + [1...M-1 remaining]
    # Let's re-evaluate Sample 4: N=3, K=3. M=2.
    # [2]*3, [1]*K? No. 2 2 2 1 3 3 3 1 1
    # It is: M(K), 1(K), 3(K), 1(K) is wrong.
    # It is: 2,2,2 (M), then 1, 3,3,3, 1,1.
    # That is: M(K), then 1(1), 3(K), 1(K-1).
    # Let's look at Sample 1: N=2, K=2. M=1.
    # 1 2 2 1. That is: 1(1), 2(2), 1(1).
    # Sample 3: N=6, K=1. M=3.
    # 3 6 5 4 2 1. That is: 3(1), 6(1), 5(1), 4(1), 2(1), 1(1).
    
    # The pattern is:
    # The middle element M = (N+1)//2.
    # The sequence is:
    # M (K times), then 
    # for i from N down to 1:
    #   if i == M: continue
    #   i (K times)
    # But wait, the 1s in Sample 4 are split.
    # Sample 4: 2 2 2 | 1 | 3 3 3 | 1 1
    # This looks like: M(K), then 1(1), then N(K), N-1(K)... M+1(K), then 1(K-1), 2(K)...
    # Actually, the simplest observation for floor((S+1)/2) is:
    # It is the sequence that is "lexicographically" the middle.
    # The sequence is: 
    # For i = 1 to N:
    #   If i < (N+1)/2: place i at the end.
    #   If i > (N+1)/2: place i at the beginning.
    #   If i == (N+1)/2: place i in the middle.
    # Let's check Sample 4: N=3, K=3. M=2.
    # i=1: end. i=3: beginning. i=2: middle.
    # Result: 3 3 3 2 2 2 1 1 1. 
    # But sample says 2 2 2 1 3 3 3 1 1.
    
    # Let's use the property: the sequence is the one where we 
    # greedily pick the largest possible digit d such that 
    # the number of sequences starting with prefix + d is <= S/2.
    # The number of ways to complete a sequence is (rem_total)! / product(rem_i!)
    # This is a huge number, so we use the property that we want the 
    # "middle" sequence. The middle sequence is the one that is 
    # "self-complementary" in a sense.
    # The actual answer is:
    # For i from 1 to N:
    #   If i < (N+1)/2: it's "small"
    #   If i > (N+1)/2: it's "large"
    #   If i == (N+1)/2: it's "middle"
    # The sequence is: M(K), 1(1), (N)(K), (N-1)(K)... (M+1)(K), 1(K-1), 2(K)... (M-1)(K)
    # No, that's too complex. Let's use the symmetry:
    # The sequence is the one where we pick digit d at each step such that
    # the number of sequences starting with digits < d is < S/2
    # and the number of sequences starting with digits <= d is >= S/2.
    # Since we can't compute S, we use the fact that the "middle" sequence
    # is the one that is its own complement if we reverse the alphabet and the sequence.
    # The sequence is: 
    # For j from 0 to N*K - 1:
    #   The digit is (N+1)//2 if we are at the "middle" of the remaining counts.
    # Actually, the simplest construction is:
    # The sequence is: M(K), then (M-1)(K), ..., 1(K), then (M+1)(K), ..., N(K)
    # But reversed? 
    # Let's try: M(K), then 1(1), then N(K), N-1(K)... M+1(K), then 1(K-1), 2(K)...
    # Let's use the property: the sequence is the one that is 
    # "lexicographically" the middle.
    # For N=3, K=3, S = 9!/(3!^3) = 1680. Target = 840th.
    # The sequence is: 2 2 2 1 3 3 3 1 1.
    # This is: M(K), 1(1), N(K), N-1(K)... M+1(K), 1(K-1), 2(K)... M-1(K).
    # Wait, 1(1) then 1(2). That's just 1(K) split by the large numbers.
    # The pattern is: 
    # 1. Middle element M = (N+1)//2.
    # 2. Start with M repeated K times.
    # 3. Then 1 repeated 1 time.
    # 4. Then N, N-1, ..., M+1 each repeated K times.
    # 5. Then 1 repeated K-1 times, 2 repeated K times, ..., M-1 repeated K times.
    # Let's check Sample 1: N=2, K=2. M=1.
    # 1(2), 1(1), 2(2), 1(2-1=1). -> 1 1 1 2 2 1. Wrong.
    
    # Let's try another pattern:
    # The sequence is: M(K), then 1(1), then (M+1...N) in some order, then (1...M-1) in some order.
    # Sample 1: 1 2 2 1 (N=2, K=2, M=1). M(1), 2(2), 1(1).
    # Sample 3: 3 6 5 4 2 1 (N=6, K=1, M=3). M(1), 6(1), 5(1), 4(1), 2(1), 1(1).
    # Sample 4: 2 2 2 1 3 3 3 1 1 (N=3, K=3, M=2). M(3), 1(1), 3(3), 1(2).
    
    # The pattern is:
    # 1. M = (N+1)//2
    # 2. M repeated K times.
    # 3. 1 repeated 1 time.
    # 4. N, N-1, ..., M+1 each repeated K times.
    # 5. 1 repeated K-1 times.
    # 6. 2, 3, ..., M-1 each repeated K times.