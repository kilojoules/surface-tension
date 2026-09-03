import sys

def solve():
    # Read N and M from the first line of input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))
    
    # Let P[i] be the distance from rest area 1 to rest area i+1.
    # P[0] = 0
    # P[1] = A_1
    # P[2] = A_1 + A_2
    # ...
    # P[N] = A_1 + ... + A_N (Total circumference)
    
    P = [0] * (N + 1)
    for i in range(N):
        P[i+1] = P[i] + A[i]
    
    total_dist = P[N]
    
    # The distance from s to t (s != t) clockwise is:
    # If s < t: P[t-1] - P[s-1]
    # If s > t: (P[N] - P[s-1]) + P[t-1]
    
    # We want distance % M == 0.
    # Let X_i = P[i-1] % M for i = 1 to N.
    # If s < t: (X_t - X_s) % M == 0  => X_t == X_s
    # If s > t: (total_dist - X_s + X_t) % M == 0 => (X_s - X_t) % M == total_dist % M
    
    # Let R = total_dist % M.
    # Condition for s < t: X_s == X_t
    # Condition for s > t: X_s - X_t ≡ R (mod M) => X_t ≡ (X_s - R) (mod M)
    
    X = [P[i] % M for i in range(N)]
    
    # Count occurrences of each remainder
    counts = {}
    for val in X:
        counts[val] = counts.get(val, 0) + 1
    
    ans = 0
    R = total_dist % M
    
    # For each s, we look for t such that the condition is met.
    # To avoid double counting and handle s < t vs s > t:
    # Total pairs (s, t) is sum over all s of (count of t such that dist(s,t) % M == 0)
    # minus cases where s == t (though the problem says s != t).
    
    # For a fixed s, t is valid if:
    # 1. s < t and X[t-1] == X[s-1]
    # 2. s > t and X[t-1] == (X[s-1] - R) % M
    
    # This is equivalent to:
    # For every pair (s, t) with s < t:
    #   Check if X[s-1] == X[t-1]
    # For every pair (s, t) with s > t:
    #   Check if X[t-1] == (X[s-1] - R) % M
    
    # Let's use the counts.
    # For s < t, the number of pairs is sum(count[v] * (count[v] - 1) // 2)
    for v in counts:
        c = counts[v]
        ans += c * (c - 1) // 2
        
    # For s > t, we need X[t-1] == (X[s-1] - R) % M.
    # This is sum_{s=1 to N} (count of t < s such that X[t-1] == (X[s-1] - R) % M)
    # We can iterate through the array X and keep track of counts of elements seen so far.
    
    # However, the "s > t" logic is simpler:
    # For every s, we need a t < s such that X[t-1] == (X[s-1] - R) % M.
    # Let's track counts of X values as we iterate.
    
    # Reset ans for s > t part and calculate carefully.
    # The s < t part is already done.
    
    # For s > t:
    # We need X[t-1] = (X[s-1] - R) % M.
    # Let's iterate s from 1 to N. For each s, the number of valid t's is the 
    # number of times (X[s-1] - R) % M has appeared in X[0...s-2].
    
    # But wait, the s < t part was: X[s-1] == X[t-1].
    # Let's just iterate once.
    
    # Re-calculating everything in one pass:
    final_ans = 0
    current_counts = {}
    for i in range(N):
        val_s = X[i]
        # This i is acting as 't' for s < t
        # We need X[s-1] == X[t-1] where s-1 < i
        final_ans += current_counts.get(val_s, 0)
        
        # This i is acting as 's' for s > t
        # We need X[t-1] == (X[s-1] - R) % M where t-1 < i
        target_t = (val_s - R) % M
        final_ans += current_counts.get(target_t, 0)
        
        # Special case: if R == 0, then (X[s-1] - R) % M == X[s-1].
        # The logic above counts t < s twice if R == 0.
        # But the problem says s != t. 
        # If R == 0, then dist(s, t) % M == 0 and dist(t, s) % M == 0.
        # The loop above:
        # When i is t: it adds count of s < t where X[s] == X[t].
        # When i is s: it adds count of t < s where X[t] == (X[s] - 0) % M.
        # These are the same pairs, just different roles.
        # Actually, if R == 0, then for any pair {s, t}, both (s,t) and (t,s) are valid.
        # The loop correctly adds 1 for (s,t) and 1 for (t,s).
        
        # Wait, if R == 0, the target_t is the same as val_s.
        # The line `final_ans += current_counts.get(val_s, 0)` handles s < t.
        # The line `final_ans += current_counts.get(target_t, 0)` handles s > t.
        # If R == 0, both lines add the same value. This is correct because both (s,t) and (t,s) are valid.
        # BUT, there is a catch: the loop processes i as the second element of the pair.
        # For a pair {s, t} with s < t:
        # i = t: `final_ans += current_counts.get(X[t], 0)` -> counts pair (s, t)
        # i = t: `final_ans += current_counts.get((X[t]-R)%M, 0)` -> counts pair (t, s) if (X[t]-R)%M == X[s]
        
        # Let's trace Sample 1: N=4, M=3, A=[2, 1, 4, 3]
        # P = [0, 2, 3, 7, 10], X = [0, 2, 0, 1], total_dist = 10, R = 1
        # i=0: X[0]=0. counts={0:1}, ans=0
        # i=1: X[1]=2. s<t: X[s]=2 (0), s>t: X[t]=(2-1)%3=1 (0). counts={0:1, 2:1}, ans=0
        # i=2: X[2]=0. s<t: X[s]=0 (1), s>t: X[t]=(0-1)%3=2 (1). counts={0:2, 2:1}, ans=2
        # i=3: X[3]=1. s<t: X[s]=1 (0), s>t: X[t]=(1-1)%3=0 (2). counts={0:2, 2:1, 1:1}, ans=2+2=4
        # Result: 4. Correct.
        
        current_counts[val_s] = current_counts.get(val_s, 0) + 1

    print(final_ans)

if __name__ == "__main__":
    solve()