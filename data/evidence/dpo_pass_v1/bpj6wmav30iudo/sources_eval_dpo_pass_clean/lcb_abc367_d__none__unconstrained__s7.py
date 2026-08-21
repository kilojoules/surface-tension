import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Parse N and M
    N = int(input_data[0])
    M = int(input_data[1])
    # Parse A_i list
    A = list(map(int, input_data[2:]))
    
    # Let S[i] be the distance from rest area 1 to rest area i+1.
    # S[0] = 0
    # S[1] = A[0]
    # S[2] = A[0] + A[1] ...
    # The distance from s to t (s < t) is S[t-1] - S[s-1].
    # The distance from s to t (s > t) is (Total_Sum - S[s-1]) + S[t-1].
    
    # Generate prefix sums: S[0], S[1], ..., S[N-1]
    # We use accumulate to avoid for-loops.
    # We only need the sums modulo M.
    S = list(accumulate(A[:-1], lambda x, y: (x + y) % M, initial=0))
    
    # Total sum of all A_i modulo M
    total_sum = sum(A) % M
    
    # Count frequencies of each remainder in S
    # S contains remainders for indices 1 to N
    # Note: S has N elements (0 to N-1)
    counts = Counter(S)
    
    # For a fixed s and t (s != t):
    # If s < t: distance is (S[t-1] - S[s-1]) % M == 0  => S[t-1] == S[s-1]
    # If s > t: distance is (total_sum - S[s-1] + S[t-1]) % M == 0 => S[s-1] - S[t-1] == total_sum
    
    # Case 1: s < t. For each remainder r, if it appears C times, 
    # there are C * (C - 1) // 2 pairs.
    # We use a generator expression inside sum() to avoid loops.
    ans_st = sum(C * (C - 1) // 2 for C in counts.values())
    
    # Case 2: s > t. We need S[s-1] - S[t-1] ≡ total_sum (mod M).
    # This is equivalent to S[s-1] ≡ S[t-1] + total_sum (mod M).
    # For each r, we look for r + total_sum.
    # To avoid double counting and handle s > t correctly:
    # We iterate through the unique remainders present in the set.
    # If total_sum == 0, this is the same as Case 1.
    # If total_sum != 0, we count pairs (r, (r + total_sum) % M).
    
    # We can calculate Case 2 by summing counts[r] * counts[(r + total_sum) % M]
    # But we must be careful: the condition s > t is strict.
    # Let's use the property: 
    # Total pairs (s, t) such that dist(s, t) % M == 0 is:
    # sum_{s=1 to N} (count of t such that dist(s, t) % M == 0)
    
    # For a fixed s, we want t such that:
    # 1. t > s and S[t-1] ≡ S[s-1] (mod M)
    # 2. t < s and S[t-1] ≡ S[s-1] - total_sum (mod, M)
    
    # Let's rethink:
    # For every pair {i, j} with i < j:
    # Clockwise i -> j is (S[j-1] - S[i-1]) % M
    # Clockwise j -> i is (total_sum - (S[j-1] - S[i-1])) % M
    
    # If S[j-1] == S[i-1], then i -> j is 0 mod M.
    # If S[j-1] - S[i-1] == total_sum, then j -> i is 0 mod M.
    
    # Let C[r] be the number of times remainder r appears in S.
    # Pairs (i, j) with i < j such that S[i-1] == S[j-1] is C[r]*(C[r]-1)//2.
    # Pairs (i, j) with i < j such that S[j-1] - S[i-1] == total_sum is C[r] * C[(r + total_sum) % M].
    # Special case: if total_sum == 0, the second condition is the same as the first.
    # But the problem asks for s != t.
    
    # If total_sum == 0:
    # s -> t is 0 mod M iff S[t-1] == S[s-1].
    # For each r, there are C[r] choices for s and C[r]-1 choices for t.
    # Total = sum( C[r] * (C[r] - 1) )
    
    # If total_sum != 0:
    # s < t: S[t-1] == S[s-1]  => sum( C[r] * (C[r] - 1) // 2 )
    # s > t: S[s-1] - S[t-1] == total_sum => sum( C[r] * C[(r - total_sum) % M] )
    # Wait, if s > t, distance is (total_sum - (S[s-1] - S[t-1])) % M.
    # For this to be 0, S[s-1] - S[t-1] must be total_sum % M.
    # So S[s-1] ≡ S[t-1] + total_sum (mod M).
    
    # Let't use a unified approach:
    # For every pair i < j:
    # Check if (S[j-1] - S[i-1]) % M == 0  (this is s=i, t=j)
    # Check if (total_sum - (S[j-1] - S[i-1])) % M == 0 (this is s=j, t=i)
    
    # result = sum(C[r]*(C[r]-1)//2) + sum(C[r] * C[(r + total_sum) % M])
    # But if total_sum == 0, the second term is sum(C[r]^2). 
    # That's not quite right. Let's use the logic:
    # If total_sum == 0:
    #   ans = sum( C[r] * (C[r] - 1) )
    # If total_sum != 0:
    #   ans = sum( C[r] * (C[r] - 1) // 2 ) + sum( C[r] * C[(r + total_sum) % M] )
    # Wait, the second term: for a fixed r, we have C[r] indices where S is r,
    # and C[(r + total_sum)%M] indices where S is (r + total_sum)%M.
    # Any pair (s, t) with s > t where S[s-1] is (r + total_sum)%M and S[t-1] is r
    # will satisfy the condition.
    # So for each r, we add C[(r + total_sum)%M] * C[r].
    
    # Correct Logic:
    # Let C = Counter of S.
    # Part 1 (s < t): sum( C[r] * (C[r] - 1) // 2 )
    # Part 2 (s > t): sum( C[r] * C[(r - total_sum) % M] ) 
    # Note: in Part 2, r is S[s-1] and (r - total_sum) is S[t-1].
    # Since s > t, we don't need to worry about the C(n,2) formula, 
    # we just need the product of counts. 
    # However, we must ensure we don't count the same pair if total_sum == 0.
    
    # If total_sum == 0:
    # s < t: S[t-1] == S[s-1]
    # s > t: S[s-1] == S[t-1]
    # Total = sum( C[r] * (C[r] - 1) )
    
    # If total_sum != 0:
    # s < t: S[t-1] == S[s-1]
    # s > t: S[s-1] - S[t-1] == total_sum
    # Total = sum( C[r] * (C[r] - 1) // 2 ) + sum( C[r] * C[(r - total_sum) % M] )
    
    # To implement without loops:
    # Use the keys of the counter.
    
    # Part 1
    p1 = sum(C * (C - 1) // 2 for C in counts.values())
    
    # Part 2
    # We use a generator to iterate over the unique remainders found in S
    p2 = sum(counts[r] * counts[(r - total_sum) % M] for r in counts)
    
    # If total_sum is 0, p2 counts pairs where S[s-1] == S[t-1] for s > t.
    # That is exactly the same as p1.
    # So if total_sum == 0, ans = p1 + p1 = sum(C*(C-1))
    # If total_sum != 0, ans = p1 + p2.
    
    # But wait, if total_sum == 0, then (r - total_sum) % M == r.
    # Then p2 = sum( counts[r] * counts[r] ).
    # That's not quite C*(C-1)//2.
    # Let's use a conditional.
    
    final_ans = (sum(C * (C - 1) for C in counts.values()) 
                 if total_sum == 0 
                 else p1 + p2)
    
    sys.stdout.write(str(final_ans) + '\n')

if __name__ == '__main__':
    solve()