import sys
from functools import reduce

def solve():
    # Read all input at once and split into a list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract N and M, then the remaining A_i
    def get_params(data):
        n = next(data)
        m = next(data)
        return n, m, data

    n, m, a_gen = get_params(input_data)
    
    # Calculate prefix sums modulo M.
    # P_i is the distance from rest area 1 to rest area i+1.
    # P_0 = 0
    # P_1 = A_1 % M
    # P_2 = (A_1 + A_2) % M ...
    # We use reduce to build the list of prefix sums.
    # The accumulator is (current_sum, list_of_sums).
    prefix_sums = reduce(
        lambda acc, x: ( (acc[0] + x) % m, acc[1] + [(acc[0] + x) % m] ),
        a_gen,
        (0, [0])
    )[1]

    # The distance from s to t (s < t) is (P_{t-1} - P_{s-1}) % M.
    # This is 0 mod M if P_{t-1} == P_{s-1}.
    # The distance from s to t (s > t) is (Total_Sum - P_{s-1} + P_{t-1}) % M.
    # This is 0 mod M if (P_{s-1} - P_{t-1}) % M == Total_Sum % M.
    
    total_sum_mod_m = prefix_sums[-1]
    
    # We need to count pairs (s, t) such that:
    # 1. s < t and P_{t-1} == P_{s-1} (mod M)
    # 2. s > t and (P_{s-1} - P_{t-1}) == Total_Sum (mod M)
    
    # Let's count occurrences of each remainder in prefix_sums (excluding the last one 
    # because the problem says s != t, and the distance from 1 to N+1 is the total sum).
    # Actually, the prefix sums are P_0, P_1, ..., P_{N-1}.
    # P_i is the distance from area 1 to area i+1.
    # There are N such points.
    
    # Use a dictionary to count frequencies of each remainder
    counts = reduce(
        lambda acc, x: acc.update({x: acc.get(x, 0) + 1}) or acc,
        prefix_sums[:-1],
        {}
    )
    
    # For a fixed remainder r, let c = counts[r].
    # Pairs (s, t) with s < t contributing to the sum: c * (c - 1) // 2
    # Pairs (s, t) with s > t:
    # We need P_{s-1} - P_{t-1} \equiv Total_Sum (mod M)
    # Let P_{t-1} = r, then P_{s-1} = (r + Total_Sum) % M.
    # The number of such pairs is counts[r] * counts[(r + Total_Sum) % M].
    # Special case: if Total_Sum % M == 0, then s > t is also P_{s-1} == P_{t-1}.
    # But we must ensure s != t.
    
    # To avoid loops, we use the keys of the counts dictionary and sum.
    # We use a generator expression inside sum().
    
    # Part 1: s < t
    ans_lt = sum(c * (c - 1) // 2 for c in counts.values())
    
    # Part 2: s > t
    # If Total_Sum % M == 0, then s > t is the same condition as s < t.
    # If Total_Sum % M != 0, then we count pairs (r, (r + Total_Sum) % M).
    if total_sum_mod_m == 0:
        ans_gt = sum(c * (c - 1) // 2 for c in counts.values())
    else:
        ans_gt = sum(counts[r] * counts.get((r + total_sum_mod_m) % m, 0) 
                     for r in counts)
    
    print(ans_lt + ans_gt)

if __name__ == "__main__":
    solve()