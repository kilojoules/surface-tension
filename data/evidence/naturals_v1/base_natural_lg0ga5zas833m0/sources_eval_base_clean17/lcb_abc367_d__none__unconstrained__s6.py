```python
import sys

def solve():
    # Read N and M from the first line
    try:
        line1 = sys.stdin.readline().split()
        if not line1:
            return
        n, m = map(int, line1)
        
        # Read A_i from the following lines
        a = []
        while len(a) < n:
            line = sys.stdin.readline().split()
            if not line:
                break
            a.extend(map(int, line))
    except ValueError:
        return

    # Let P[i] be the distance from rest area 1 to rest area i clockwise.
    # P[1] = 0
    # P[2] = A_1
    # P[3] = A_1 + A_2
    # ...
    # P[N] = A_1 + ... + A_{N-1}
    # Total distance L = A_1 + ... + A_N
    
    p = [0] * (n + 1)
    for i in range(n):
        p[i+1] = p[i] + a[i]
    
    total_dist = p[n]
    
    # We want to find pairs (s, t) such that the clockwise distance from s to t is 0 mod M.
    # If s < t: dist(s, t) = P[t] - P[s]
    # If s > t: dist(s, t) = (Total - P[s]) + P[t] = Total + P[t] - P[s]
    
    # Let x_i = P[i] mod M for i = 1 to N.
    # Case 1: s < t
    # P[t] - P[s] ≡ 0 (mod M)  =>  x_t ≡ x_s (mod M)
    # Case 2: s > t
    # Total + P[t] - P[s] ≡ 0 (mod M)  =>  x_s - x_t ≡ Total (mod M)
    
    # Let's count frequencies of each remainder x_i
    count = [0] * m
    for i in range(1, n + 1):
        count[p[i-1] % m] += 1
    # Note: P[0] is distance to area 1, P[1] to area 2... P[N-1] to area N.
    # The loop above uses p[0]...p[n-1].
    
    # Correcting the indices:
    # Rest areas are 1, ..., N.
    # Dist from 1 to 1 is 0.
    # Dist from 1 to 2 is A_1.
    # Dist from 1 to i is P[i-1].
    
    # Let's redefine:
    # S_i = distance from area 1 to area i.
    # S_1 = 0
    # S_2 = A_1
    # S_3 = A_1 + A_2
    # ...
    # S_N = A_1 + ... + A_{N-1}
    # Total L = A_1 + ... + A_N
    
    # For s < t: (S_t - S_s) % M == 0  =>  S_t % M == S_s % M
    # For s > t: (L + S_t - S_s) % M == 0  =>  S_s - S_t % M == L % M
    
    # Precalculate S_i % M
    s_mod = [0] * n
    current_sum = 0
    for i in range(n - 1):
        current_sum += a[i]
        s_mod[i+1] = current_sum % m
    
    # Frequency of each remainder
    freq = [0] * m
    for val in s_mod:
        freq[val] += 1
        
    ans = 0
    l_mod = total_dist % m
    
    # Case 1: s < t. For each remainder r, there are freq[r] items.
    # The number of pairs (s, t) with s < t and S_s % M == S_t % M is freq[r] * (freq[r] - 1) // 2.
    # However, we need to consider s and t specifically. 
    # If we have freq[r] indices with the same remainder, there are C(freq[r], 2) pairs.
    for r in range(m):
        ans += freq[r] * (freq[r] - 1) // 2
        
    # Case 2: s > t.
    # We need S_s - S_t ≡ L (mod M)  =>  S_s ≡ (L + S_t) (mod M)
    # For each t, we need to count s such that s > t and S_s ≡ (L + S_t) (mod M).
    # This is tricky because s > t. Let's use a different approach.
    
    # Total pairs (s, t) with s != t is N * (N - 1).
    # For a fixed s and t:
    # If s < t, condition is S_t - S_s ≡ 0 (mod M)
    # If s > t, condition is L + S_t - S_s ≡ 0 (mod M)
    
    # Let's iterate through all possible remainders r1 and r2.
    # If S_s % M = r1 and S_t % M = r2:
    # If s < t, we need r2 - r1 ≡ 0 (mod M) => r1 == r2.
    # If s > t, we need L + r2 - r1 ≡ 0 (mod M) => r1 - r2 ≡ L (mod M).
    
    # Let's re-evaluate.
    # For any pair {s, t} with s < t:
    # One direction is s -> t (dist S_t - S_s)
    # Other direction is t -> s (dist L + S_s - S_t)
    
    # For s < t:
    # s -> t is a multiple of M if S_t ≡ S_s (mod M)
    # t -> s is a multiple of M if S_s - S_t ≡ -L (mod M) => S_t - S_s ≡ L (mod M)
    
    # Let r_s = S_s % M and r_t = S_t % M.
    # For each pair {s, t} with s < t:
    # - It's a valid pair (s, t) if r_t - r_s ≡ 0 (mod M)
    # - It's a valid pair (t, s) if r_s - r_t ≡ L (mod M) => r_t - r_s ≡ -L (mod M)
    
    # Total valid pairs = Sum_{s < t} [r_t - r_s ≡ 0] + Sum_{s < t} [r_t - r_s ≡ -L]
    
    # Sum_{s < t} [r_t - r_s ≡ 0] is simply the number of pairs with same remainder.
    # Sum_{s < t} [r_t - r_s ≡ -L] is the number of pairs where r_t - r_s ≡ (M - L % M) % M.
    
    # Let target = (m - l_mod) % m
    # If target == 0:
    #   The two conditions are the same: r_t - r_s ≡ 0.
    #   But we must be careful. If target == 0, then L is a multiple of M.
    #   Then for any s < t, if S_t ≡ S_s (mod M), both s->t and t->s are multiples of M.
    #   So we count each such pair twice.
    # If target != 0:
    #   The conditions are different.
    #   Number of pairs (s, t) with s < t and r_t - r_s ≡ 0 is C(freq[r], 2).
    #   Number of pairs (s, t) with s < t and r_t - r_s ≡ target is...
    #   Wait, the target condition depends on the relative order of s and t.
    
    # Let's use the property:
    # Total = Sum_{s < t} [r_t - r_s ≡ 0] + Sum_{s < t} [r_t - r_s ≡ -L]
    # The first part is Sum (freq[r] * (freq[r]-1) // 2).
    # For the second part:
    # We need to count pairs (s, t) with s < t such that r_t - r_s ≡ target (mod M).
    # This can be done by iterating through the array s_mod and keeping track of counts.
    
    target = (m - l_mod) % m
    count_target = 0
    current_freq = [0] * m
    for val in s_mod:
        # We need (val - r_s) % m == target  =>  r_s = (val - target) % m
        count_target += current_freq[(val - target) % m]
        current_freq[val] += 1
        
    # If target == 0, the count_target is the same as the first part.
    # The logic:
    # ans = (number of pairs s < t with r_s == r_t) + (number of pairs s < t with r_t - r_s == target)
    # If target == 0, these are the same pairs, but the problem asks for (s, t) and (t, s).
    # Sample 1: N=4, M=3, A=[2, 1, 4, 3]. L=10, L%3=1. target=(3-1)%3=2.
    # S = [0, 2, 3, 7] -> S%3 = [0, 2, 0, 1]
    # Pairs s < t with r_s == r_t: (0, 2) -> 1 pair.
    # Pairs s < t with r_t - r_s == 2:
    # val=0: current_freq[0]=1
    # val=2: target=2, r_s=(2-2)%3=0. current_freq[0]=1. count=1. current_freq[2]=1.
    # val=0: target=2, r_s=(0-2)%3=1. current_freq[1]=0. count=1. current_freq[0]=2.
    # val=1: target=2, r_s=(1-2)%3=2. current_freq[2]=1. count=2. current_freq[1]=1.
    # Total = 1 + 2 = 3.
    # Wait, Sample 1 output is 4. Let me re-read.
    # S = [0, 2, 3, 7] is wrong.
    # S_1 = 0
    # S_2 = A_1 = 2
    # S_3 = A_1 + A_2 = 2 + 1 = 3
    # S_4 = A_1 + A_2 + A_3 = 2 + 1 + 4 = 7
    # S_mod = [0, 2, 0, 1]
    # s < t:
    # (1,2): 2-0=2 (no)
    # (1,3): 0-0=0 (yes)
    # (1,4): 1-0=1 (no)
    # (2,3): 0-2=-2=1 (no)
    # (2,4): 1-2=-1=2 (no)
    # (3,4): 1-0=1 (no)
    # s > t:
    # (2,1): 10+0-2=8 (no)
    # (3,1): 10+0-0=10 (no)
    # (3,2): 10+2-0=12 (yes)
    # (4,1): 10+0-1=9 (yes)
    # (4,2): 10+2-1=11 (no)
    # (4,3): 10+0-1=9 (yes)
    # Total = 1 + 3 = 4. Correct.
    
    # My manual calculation for count_target:
    # S