```python
import sys
from functools import reduce

def solve():
    # Read N and M
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))

    # Calculate prefix sums modulo M
    # P[i] is the distance from rest area 1 to rest area i+1
    # P[0] = 0, P[1] = A[0], P[2] = A[0] + A[1], ...
    # We use reduce to build the prefix sum list without explicit loops
    # The prefix sums are taken modulo M
    P = list(reduce(lambda acc, x: acc + [ (acc[-1] + x) % M ], [0], A))
    
    # The distance from s to t (s < t) is (P[t-1] - P[s-1]) % M
    # We want (P[t-1] - P[s-1]) % M == 0, which means P[t-1] == P[s-1]
    # The distance from s to t (s > t) is (TotalSum - P[s-1] + P[t-1]) % M
    # We want (TotalSum - P[s-1] + P[t-1]) % M == 0
    # Which means P[s-1] - P[t-1] == TotalSum % M
    
    # We only need the first N prefix sums (P[0] to P[N-1])
    # P[N] is the total sum modulo M
    prefix_sums = P[:N]
    total_sum_mod = P[N]
    
    # Count occurrences of each remainder modulo M
    # Using a list as a frequency array since M <= 10^6
    counts = [0] * M
    for x in prefix_sums:
        counts[x] += 1
        
    # For each remainder r, let c = counts[r].
    # Pairs (s, t) with s < t and P[s-1] == P[t-1] == r: c * (c - 1) // 2
    # Pairs (s, t) with s > t and P[s-1] - P[t-1] == total_sum_mod:
    # This is equivalent to P[s-1] == (P[t-1] + total_sum_mod) % M
    # For a fixed r = P[t-1], we need P[s-1] = (r + total_sum_mod) % M
    # The number of such pairs is counts[r] * counts[(r + total_sum_mod) % M]
    # However, we must exclude the case where s = t (though the problem says s != t)
    # and handle the case where total_sum_mod == 0 carefully.
    
    # If total_sum_mod == 0, the two conditions merge into P[s-1] == P[t-1].
    # There are N * (N-1) pairs if we just look at remainders, but we must 
    # group by remainder: sum(c * (c-1))
    
    # If total_sum_mod != 0:
    # Pairs s < t: sum(c * (c-1) // 2)
    # Pairs s > t: sum(counts[r] * counts[(r + total_sum_mod) % M])
    # Wait, the s > t logic: 
    # Dist(s, t) = (Total - P[s-1]) + P[t-1]
    # (Total + P[t-1] - P[s-1]) % M == 0  =>  P[s-1] == (Total + P[t-1]) % M
    
    # Let's use a more general approach:
    # For every pair (s, t) with s != t:
    # If s < t: condition is P[t-1] == P[s-1]
    # If s > t: condition is P[s-1] == (P[t-1] + total_sum_mod) % M
    
    # Total = sum_{r=0 to M-1} [ (counts[r]*(counts[r]-1)//2) + (counts[r]*counts[(r+total_sum_mod)%M]) ]
    # But if total_sum_mod == 0, the second term counts pairs where P[s-1] == P[t-1]
    # and s > t. So it becomes sum(c*(c-1)//2 + c*(c-1))? No.
    # If total_sum_mod == 0, then P[s-1] == P[t-1] is the only condition.
    # There are c*(c-1) pairs for each remainder.
    
    # Correct logic:
    # For each r in 0...M-1:
    #   Ways s < t: counts[r] * (counts[r] - 1) // 2
    #   Ways s > t: counts[(r + total_sum_mod) % M] * counts[r] 
    #   BUT if total_sum_mod == 0, the s > t case is just counts[r] * (counts[r] - 1)
    #   Wait, if total_sum_mod == 0, then s > t and P[s-1] == P[t-1] is the condition.
    #   That is also c * (c-1) // 2.
    #   Total for total_sum_mod == 0 is sum(c * (c-1))
    
    # Let's refine:
    # If total_sum_mod == 0:
    #    Ans = sum(c * (c - 1))
    # Else:
    #    Ans = sum(c * (c - 1) // 2) + sum(counts[r] * counts[(r + total_sum_mod) % M])
    #    Wait, the second sum is over all r. 
    #    For a fixed r, counts[r] is the number of t's, and counts[(r + total_sum_mod)%M] is the number of s's.
    #    Since total_sum_mod != 0, the sets of indices are disjoint, so s != t is guaranteed.
    
    # Let's double check Sample 1: N=4, M=3, A=[2, 1, 4, 3]
    # P = [0, 2, 0, 1, 1] -> prefix_sums = [0, 2, 0, 1], total_sum_mod = 1
    # counts: {0: 2, 1: 1, 2: 1}
    # s < t: (2*1//2) + (1*0//2) + (1*0//2) = 1
    # s > t: r=0: c[0]*c[1] = 2*1 = 2; r=1: c[1]*c[2] = 1*1 = 1; r=2: c[2]*c[0] = 1*2 = 2
    # Total = 1 + (2 + 1 + 2) = 6? Sample says 4.
    # Let's re-read: "minimum number of steps to walk clockwise from s to t"
    # s=1, t=3: P[2]-P[0] = 0-0 = 0 (mod 3). Correct. (s < t)
    # s=3, t=2: (Total - P[2]) + P[1] = (1 - 0) + 2 = 3 = 0 (mod 3). Correct. (s > t)
    # s=4, t=1: (Total - P[3]) + P[0] = (1 - 1) + 0 = 0 (mod 3). Correct. (s > t)
    # s=4, t=3: (Total - P[3]) + P[2] = (1 - 1) + 0 = 0 (mod 3). Correct. (s > t)
    # Total = 4.
    
    # My manual trace:
    # s < t: P[t-1] == P[s-1]. Pairs: (1,3) since P[0]=0, P[2]=0. Count = 1.
    # s > t: P[s-1] == (P[t-1] + total_sum_mod) % M.
    # r=0 (t=1 or 3): P[s-1] == (0 + 1) % 3 = 1. s=4 (P[3]=1). Pairs: (4,1), (4,3). Count = 2.
    # r=1 (t=4): P[s-1] == (1 + 1) % 3 = 2. s=2 (P[1]=2). Pairs: (2,4). Count = 1.
    # r=2 (t=2): P[s-1] == (2 + 1) % 3 = 0. s=1 or 3 (P[0]=0, P[2]=0). Pairs: (1,2), (3,2). Count = 2.
    # Total = 1 + 2 + 1 + 2 = 6. Still 6. What's wrong?
    # "minimum number of steps to walk clockwise from s to t (s != t)"
    # If s=1, t=2: steps = A1 = 2. 2%3 != 0.
    # If s=3, t=2: steps = A3 + A4 + A1 = 4 + 3 + 2 = 9. 9%3 == 0.
    # My formula for s > t: Dist = (TotalSum - P[s-1]) + P[t-1]
    # TotalSum = 2+1+4+3 = 10. P = [0, 2, 3, 7, 10]
    # s=3, t=2: (10 - P[2]) + P[1] = (10 - 3) + 2 = 9. Correct.
    # My modulo arithmetic: (10 - 3 + 2) % 3 = (1 - 0 + 2) % 3 = 0.
    # Wait, the condition is (TotalSum - P[s-1] + P[t-1]) % M == 0
    # This is P[s-1] == (TotalSum + P[t-1]) % M.
    # In Sample 1: TotalSum % 3 = 1.
    # P[s-1] == (1 + P[t-1]) % 3.
    # t=1 (P=0) -> P[s-1]=1 -> s=4 (P=1). Pair (4,1).
    # t=2 (P=2) -> P[s-1]=0 -> s=1,3 (P=0). Pairs (1,2), (3,2).
    # t=3 (P=0) -> P[s-1]=1 -> s=4 (P=1). Pair (4,3).
    # t=4 (P=1) -> P[s-1]=2 -> s=2 (P=2). Pair (2,4).
    # Total s > t: 1 + 2 + 1 + 1 = 5.
    # Total s < t: P[t-1] == P[s-1] -> (1,3). Count = 1.
    # Total = 6. Still 6. Let me re-read the sample.
    # Sample 1 says: (1,3), (3,2), (4,1), (4,3). That's 4.
    # My (1,2) and (2,4) are wrong.
    # (1,2): s=1, t=2. s < t. P[1]-P[0] = 2-0 = 2. 2%3 != 0.
    # (3,2): s=3, t=2. s > t. (10-3)+2 = 9. 9%3 == 0. Correct.
    # (2,4): s=2, t=4. s < t. P[3]-P[1] = 7-2 = 5. 5%3 != 0.
    # Ah! The condition s > t is DIFFERENT.
    # If s < t, dist is P[t-1] - P[s-1].
    # If s > t, dist is (TotalSum - P[s-1]) + P[