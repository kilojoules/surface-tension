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

    # Let P_i be the prefix sum of A.
    # P_0 = 0
    # P_i = A_1 + ... + A_i
    # Distance from s to t (s < t) is (P_{t-1} - P_{s-1}) mod M == 0
    # Distance from s to t (s > t) is (P_N - P_{s-1} + P_{t-1}) mod M == 0
    
    # Calculate prefix sums modulo M
    # P[i] represents the distance from rest area 1 to rest area i+1
    P = list(accumulate(A, lambda x, y: (x + y) % M, initial=0))
    # P has N+1 elements: P[0]=0, P[1]=A1%M, ..., P[N]=(A1+...+AN)%M
    
    # We are interested in pairs (s, t) where 1 <= s, t <= N and s != t.
    # Let x = P[s-1] and y = P[t-1].
    # If s < t: (y - x) % M == 0  => y % M == x % M
    # If s > t: (P[N] - x + y) % M == 0 => (x - y) % M == P[N] % M
    
    # Let's count occurrences of each remainder in P[0...N-1]
    # Note: P[N] is the total loop length.
    # We only care about starting points s in {1...N}, so indices 0...N-1 of P.
    
    # Use a list for counting since M <= 10^6
    counts = [0] * M
    for i in range(N):
        counts[P[i]] += 1
    
    total_dist_mod = P[N]
    
    # For a fixed s, we need t such that:
    # 1. t > s and P[t-1] == P[s-1] (mod M)
    # 2. t < s and P[t-1] == (P[s-1] - total_dist_mod) (mod M)
    
    # This is equivalent to:
    # For each remainder r, there are counts[r] positions.
    # Pairs (s, t) with s < t and P[s-1] == P[t-1] == r:
    # There are counts[r] * (counts[r] - 1) // 2 such pairs.
    # However, the condition s < t is specific to the clockwise distance.
    # Let's re-evaluate:
    # A pair (s, t) is valid if:
    # If s < t: P[t-1] \equiv P[s-1] (mod M)
    # If s > t: P[t-1] \equiv P[s-1] - P[N] (mod M)
    
    # Let's use the property:
    # Total valid pairs = \sum_{s=1}^N (count of t > s where P[t-1] == P[s-1]) 
    #                   + \sum_{s=1}^N (count of t < s where P[t-1] == P[s-1] - P[N])
    
    # Let C(r) be the number of times remainder r appears in P[0...N-1].
    # The number of pairs (s, t) with s < t and P[s-1] == P[t-1] is \sum C(r)(C(r)-1)/2.
    # The number of pairs (s, t) with s > t and P[t-1] == P[s-1] - P[N] is:
    # For a fixed r, if P[s-1] = r, then we need P[t-1] = (r - P[N]) % M.
    # This looks like we can just iterate over all r and multiply C(r) * C((r - P[N]) % M).
    # But we must exclude the case where s = t (which is already excluded by s > t).
    # Wait, the s > t condition is simply: for every s, we need t < s such that P[t-1] == (P[s-1] - P[N]) % M.
    
    # Let's use a different approach:
    # For every pair of indices i, j in {0, ..., N-1} with i < j:
    # Pair (s=i+1, t=j+1) is valid if P[j] - P[i] \equiv 0 (mod M)
    # Pair (s=j+1, t=i+1) is valid if P[N] - P[j] + P[i] \equiv 0 (mod M)
    
    # Part 1: i < j and P[j] == P[i] (mod M)
    # For each r, if it appears C(r) times, there are C(r)*(C(r)-1)//2 pairs.
    
    # Part 2: i < j and P[i] == (P[j] - P[N]) (mod M)
    # This is equivalent to P[j] - P[i] == P[N] (mod M).
    # Let's iterate through the array and maintain counts of P[i] seen so far.
    # For a fixed j, we need count of i < j such that P[i] == (P[j] - P[N]) % M.
    
    # Since we cannot use loops, we can use a generator expression with sum().
    # But we need the counts of P[i] for i < j. That's hard without a loop.
    # Actually, we can use the total counts:
    # For a fixed r1 and r2 such that (r2 - r1) % M == P[N] % M:
    # We want number of pairs (i, j) such that i < j, P[i]=r1, P[j]=r2.
    # This is not simply C(r1)*C(r2) because of the i < j constraint.
    
    # Let's reconsider:
    # A pair (s, t) is valid if:
    # 1. s < t and P[t-1] \equiv P[s-1] (mod M)
    # 2. s > t and P[t-1] \equiv P[s-1] - P[N] (mod M)
    
    # Let's use the fact that we can process the array and update counts.
    # But we can't use loops. We can use a trick with a list and a function.
    # However, the most reliable way to do this without a loop in Python 
    # is to use a list comprehension that updates a state, but that's hacky.
    # Alternatively, we can use the mathematical property:
    # Total = \sum_{i < j} [P[j] == P[i]] + \sum_{i < j} [P[i] == (P[j] - P[N]) % M]
    
    # Let's use the property: 
    # \sum_{i < j} [P[i] == r1 and P[j] == r2] 
    # If r1 == r2, it's C(r1)*(C(r1)-1)//2.
    # If r1 != r2, we can't know without the positions.
    
    # Wait! The constraints on M are 10^6 and N is 2*10^5.
    # We can use a list to store the indices of each remainder.
    # indices = [ [i for i, val in enumerate(P_reduced) if val == r] for r in range(M) ]
    # But that's O(M*N).
    
    # Correct approach using list comprehension to simulate a loop:
    # We can use a list to store counts and a list comprehension to iterate.
    # Since we can't use 'for' loops, we can use 'map' or 'sum' with a generator.
    # To maintain state, we can use a mutable object (like a list).
    
    state = [0] * M
    # For each x in P[0...N-1], we want to count how many y seen so far satisfy the condition.
    # For s < t: P[t-1] == P[s-1]. When we are at index j (t-1), we need count of P[i] == P[j] for i < j.
    # For s > t: P[t-1] == P[s-1] - P[N]. When we are at index j (s-1), we need count of P[i] == (P[j] - P[N]) % M for i < j.
    
    # We can use a list comprehension to perform the summation and update the state.
    # The expression `(state.__setitem__(x, state[x] + 1), state[x])` 
    # allows us to update and return a value.
    
    # For a given P[j], the number of valid pairs ending at j is:
    # 1. t = j+1, s < t: count of P[i] == P[j] for i < j
    # 2. s = j+1, t < s: count of P[i] == (P[j] - P[N]) % M for i < j
    
    # We must be careful: if P[j] == (P[j] - P[N]) % M, we are counting the same pair?
    # No, because one is s < t and the other is s > t.
    # But if P[N] % M == 0, then P[j] == (P[j] - P[N]) % M.
    # In that case, for a fixed i < j, both (s=i+1, t=j+1) and (s=j+1, t=i+1) are valid.
    
    # To avoid loops, we use a helper function and map/sum.
    def update_and_count(x):
        # Current count of P[i] == x
        c1 = state[x]
        # Current count of P[i] == (x - total_dist_mod) % M
        c2 = state[(x - total_dist_mod) % M]
        # Update state for next elements
        state[x] += 1
        return c1 + c2

    # Use map to apply the function to all P[0...N-1]
    # Note: P[0...N-1] are the values of P[s-1]
    result = sum(map(update_and_count, P[:N]))
    print(result)

solve()