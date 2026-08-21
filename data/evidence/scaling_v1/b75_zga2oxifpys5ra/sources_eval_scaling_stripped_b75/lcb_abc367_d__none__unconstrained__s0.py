import sys
from itertools import accumulate

def solve():
    # Read N and M
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))

    # Calculate prefix sums of distances modulo M
    # P[i] is the distance from rest area 1 to rest area i+1
    # P[0] = 0, P[1] = A1 % M, P[2] = (A1 + A2) % M, ...
    # The distance from s to t (s < t) is (P[t-1] - P[s-1]) % M
    # The distance from s to t (s > t) is (Total_Sum - P[s-1] + P[t-1]) % M
    
    # We use accumulate to avoid loops for prefix sums
    # P will have N+1 elements: 0, A1, A1+A2, ..., sum(A1...AN)
    P = list(accumulate([0] + A, lambda x, y: (x + y) % M))
    
    # The total distance around the lake modulo M
    total_sum_mod = P[N]
    
    # We are looking for pairs (s, t) such that dist(s, t) % M == 0
    # For s < t: (P[t-1] - P[s-1]) % M == 0  => P[t-1] == P[s-1] (mod M)
    # For s > t: (total_sum_mod - P[s-1] + P[t-1]) % M == 0 => P[s-1] - P[t-1] == total_sum_mod (mod M)
    
    # Let's count occurrences of each remainder in P[0...N-1]
    # Note: P[N] is the total sum, but the rest areas are 1...N, 
    # so we only care about the starting positions 0...N-1.
    counts = {}
    for i in range(N):
        val = P[i]
        counts[val] = counts.get(val, 0) + 1
        
    # For a fixed remainder r, there are counts[r] positions.
    # 1. Pairs (s, t) with s < t and P[s-1] == P[t-1]:
    #    This is simply combinations(counts[r], 2)
    # 2. Pairs (s, t) with s > t and P[s-1] - P[t-1] == total_sum_mod (mod M):
    #    This is counts[r] * counts[(r - total_sum_mod) % M]
    #    HOWEVER, we must exclude the case where s == t (though the problem says s != t).
    #    If total_sum_mod == 0, then s > t and P[s-1] == P[t-1] is the same condition as s < t.
    #    If total_sum_mod != 0, the two conditions are mutually exclusive.
    
    # Using a generator expression inside sum() to avoid explicit loops
    # Part 1: s < t
    ans_lt = sum(c * (c - 1) // 2 for c in counts.values())
    
    # Part 2: s > t
    # We need P[s-1] - P[t-1] = total_sum_mod (mod M)
    # For each r, we look for r_prev = (r - total_sum_mod) % M
    ans_gt = sum(counts[r] * counts.get((r - total_sum_mod) % M, 0) 
                 for r in counts)
    
    # If total_sum_mod == 0, the s > t case counts pairs where P[s-1] == P[t-1].
    # For a group of size c, there are c*(c-1) such pairs in total (both s<t and s>t).
    # The logic above: ans_lt = c*(c-1)/2, ans_gt = c*c. 
    # Wait, if total_sum_mod == 0, then P[s-1] - P[t-1] == 0 means P[s-1] == P[t-1].
    # For every pair {i, j} with P[i]==P[j], both (i,j) and (j,i) work.
    # That is c*(c-1) total pairs.
    
    # Correct logic for total_sum_mod == 0:
    # ans = sum(c * (c - 1) for c in counts.values())
    # Correct logic for total_sum_mod != 0:
    # ans = sum(c*(c-1)//2) + sum(counts[r] * counts[(r-total)%M])
    
    if total_sum_mod == 0:
        final_ans = sum(c * (c - 1) for c in counts.values())
    else:
        final_ans = ans_lt + ans_gt
        
    print(final_ans)

if __name__ == "__main__":
    solve()