import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read N and M from the first line
    # Read A_i values from the remaining input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))
    
    # Let P_i be the distance from rest area 1 to rest area i.
    # P_1 = 0
    # P_2 = A_1
    # P_3 = A_1 + A_2
    # ...
    # P_N = A_1 + ... + A_{N-1}
    # The distance from s to t (s < t) is P_t - P_s.
    # The distance from s to t (s > t) is (Total_Sum - P_s) + P_t.
    
    # Calculate prefix sums P_1, ..., P_N
    # accumulate([0] + A[:-1]) gives P_1, ..., P_N
    # However, we need P_i mod M.
    # We use a generator to feed A into accumulate.
    # To get P_1=0, P_2=A_1, ..., P_N=sum(A_1...A_{N-1})
    # We can take the prefix sums of A and shift them.
    
    # Prefix sums of A_1...A_N
    # P = [0, A_1, A_1+A_2, ..., A_1+...+A_{N-1}]
    # We use accumulate on A and prepend 0, then take first N elements.
    prefix_sums = list(accumulate(A, lambda x, y: (x + y) % M, initial=0))[:N]
    
    # Total sum of all A_i mod M
    total_sum = sum(A) % M
    
    # Count occurrences of each remainder mod M
    counts = Counter(prefix_sums)
    
    # For a pair (s, t) with s < t:
    # Distance is (P_t - P_s) % M == 0  => P_t % M == P_s % M
    # Number of such pairs is sum(count * (count - 1) // 2)
    ans_st_lt = sum(c * (c - 1) // 2 for c in counts.values())
    
    # For a pair (s, t) with s > t:
    # Distance is (Total_Sum - P_s + P_t) % M == 0
    # => P_t % M == (P_s - Total_Sum) % M
    # We iterate over all possible remainders r = P_s % M
    # The required P_t % M is (r - total_sum) % M
    # The number of pairs is sum(counts[r] * counts[(r - total_sum) % M])
    # But we must exclude cases where s = t (though the problem says s != t, 
    # the logic s > t already handles that, but we must ensure we don't 
    # count the same index if (r - total_sum) % M == r).
    # Wait, the condition is simply: for every s, how many t < s satisfy the condition.
    # Let's use the property: Total pairs = sum_{r} (counts[r] * counts[(r - total_sum) % M])
    # This counts all (s, t) such that P_t - P_s \equiv Total_Sum (mod M)
    # This is not quite right. Let's re-evaluate.
    
    # Correct Logic:
    # Pair (s, t) is valid if:
    # 1. s < t and (P_t - P_s) % M == 0
    # 2. s > t and (Total_Sum - P_s + P_t) % M == 0
    
    # Case 1: P_t \equiv P_s (mod M) for s < t.
    # This is simply combinations of indices with same P value: nCr(count, 2).
    
    # Case 2: P_t \equiv (P_s - Total_Sum) (mod M) for s > t.
    # This is trickier because of the s > t constraint.
    # Let's use the property:
    # Total valid pairs = (Pairs where P_t - P_s \equiv 0 mod M) 
    #                   + (Pairs where P_t - P_s \equiv Total_Sum mod M)
    # Wait, that's not correct.
    
    # Let's use the fact that:
    # Distance(s, t) = (P_t - P_s) mod Total_Length
    # Clockwise distance from s to t is:
    # If s < t: P_t - P_s
    # If s > t: (Total_Sum + P_t) - P_s
    
    # We want Distance \equiv 0 (mod M).
    # If s < t: P_t \equiv P_s (mod M)
    # If s > t: P_t \equiv (P_s - Total_Sum) (mod M)
    
    # Let's iterate over all s from 1 to N.
    # For a fixed s, we need t != s such that:
    # If t > s, P_t \equiv P_s (mod M)
    # If t < s, P_t \equiv (P_s - Total_Sum) (mod M)
    
    # This can be solved by iterating through the prefix sums and maintaining a counter.
    # For each P_s:
    # 1. Add counts of P_t == (P_s - Total_Sum) % M seen so far (these are t < s)
    # 2. After the loop, we have all t < s. To get t > s, we can use the total counts.
    # Total pairs = sum_{s=1 to N} [ (count of t < s where P_t == (P_s - Total_Sum)%M) 
    #                               + (count of t > s where P_t == P_s%M) ]
    
    # Let's use a different approach:
    # Total = sum_{r=0 to M-1} (counts[r] * counts[(r - Total_Sum) % M])
    # This sum counts all pairs (s, t) such that P_t - P_s \equiv Total_Sum (mod M).
    # This is not what we want.
    
    # Let's go back to:
    # Ans = sum_{s < t} [P_t == P_s] + sum_{s > t} [P_t == (P_s - Total_Sum)]
    # Ans = sum_{r=0 to M-1} (counts[r] * (counts[r]-1) // 2) 
    #       + sum_{s=1 to N} (count of t < s such that P_t == (P_s - Total_Sum) % M)
    
    # To calculate the second term without a loop:
    # We can use a list comprehension with a side-effect (not recommended) or 
    # process the prefix sums using a custom function with reduce or a loop.
    # Since I must avoid "for" and "while", I will use a helper function with recursion 
    # or a list comprehension that updates a dictionary. 
    # Actually, the most "Pythonic" way to avoid loops while maintaining state is 
    # using a generator or map with a mutable object.
    
    # Let's use a dictionary to track counts as we iterate through prefix_sums.
    current_counts = Counter()
    def accumulate_valid_pairs(p):
        target = (p - total_sum) % M
        count = current_counts[target]
        current_counts[p] += 1
        return count

    # Use map to apply the function and sum the results
    second_term = sum(map(accumulate_valid_pairs, prefix_sums))
    
    print(ans_st_lt + second_term)

if __name__ == "__main__":
    solve()