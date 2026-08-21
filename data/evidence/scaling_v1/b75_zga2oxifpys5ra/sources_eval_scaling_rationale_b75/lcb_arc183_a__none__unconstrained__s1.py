import sys
from functools import reduce
from math import comb

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    k = int(input_data[1])

    # The total number of good sequences is S = (NK)! / (K!)^N
    # We need the floor((S+1)/2)-th sequence.
    # Since we cannot use loops or recursion, we use reduce to simulate 
    # the process of determining each element of the sequence one by one.
    
    # Helper to calculate the number of ways to arrange the remaining elements
    # Formula: (sum(counts))! / product(counts[i]!)
    # However, we only need to compare the target index with the number of 
    # sequences starting with a specific digit.
    # If we fix the first digit as 'd', the remaining ways are:
    # (TotalRemaining - 1)! / (K1! * ... * (Kd-1)! * ... * KN!)
    
    def get_count(counts):
        total = sum(counts)
        # Using a property: TotalWays = (sum(counts))! / product(c!)
        # We can't use loops, so we use reduce to calculate the denominator
        # But actually, we can use a more direct approach with math.comb
        # TotalWays = comb(total, counts[0]) * comb(total-counts[0], counts[1]) ...
        return reduce(lambda acc, c: acc * comb(total - (sum(counts[:counts.index(c)]) if counts.index(c)>0 else 0), c), 
                      range(len(counts)), 1)
    
    # The above get_count is slightly wrong because of the index() logic and range.
    # Let's redefine it using a more robust reduce.
    def get_total_permutations(counts):
        total = sum(counts)
        # Ways = total! / (c1! * c2! ... * cn!)
        # We can compute this by iteratively picking positions for each number.
        # Ways = comb(total, c1) * comb(total-c1, c2) * ...
        # We use a trick with reduce to keep track of the remaining slots.
        return reduce(lambda x, c: (x[0] - c, x[1] * comb(x[0], c)), 
                      counts, (total, 1))[1]

    # Target index: floor((S + 1) / 2)
    # S = get_total_permutations([k] * n)
    s = get_total_permutations([k] * n)
    target = (s + 1) // 2

    # State for reduce: (current_counts, current_target, result_sequence)
    # We iterate NK times (the length of the sequence)
    initial_state = ([k] * n, target, [])
    
    def step(state, _):
        counts, target_idx, seq = state
        
        # We need to find the smallest digit d (1 to N) such that 
        # the number of sequences starting with digits < d is less than target_idx,
        # and the number of sequences starting with digits <= d is >= target_idx.
        
        # To avoid loops, we use a list comprehension to calculate the number of 
        # sequences starting with each digit d in [1, N].
        # If we pick digit d, the remaining ways are:
        # ways(d) = (TotalRemaining - 1)! / (c1! ... (cd-1)! ... cn!)
        # ways(d) = TotalPermutations(counts) * counts[d-1] / TotalRemaining
        
        total_rem = sum(counts)
        total_perms = get_total_permutations(counts)
        
        # ways_for_digit[d] is the number of sequences starting with digit d+1
        ways_for_digit = [(total_perms * counts[d]) // total_rem for d in range(n)]
        
        # Find the digit d such that sum(ways_for_digit[:d]) < target_idx <= sum(ways_for_digit[:d+1])
        # We can use a list comprehension to find the first index where the cumulative sum reaches target_idx
        cum_sums = reduce(lambda acc, x: acc + [acc[-1] + x], ways_for_digit, [0])
        
        # The digit index (0-indexed) is the first i where cum_sums[i+1] >= target_idx
        digit_idx = next(i for i, s_val in enumerate(cum_sums[1:]) if s_val >= target_idx)
        
        # Update state
        new_counts = list(counts)
        new_counts[digit_idx] -= 1
        new_target = target_idx - cum_sums[digit_idx]
        
        return (new_counts, new_target, seq + [digit_idx + 1])

    final_state = reduce(step, range(n * k), initial_state)
    print(*(final_state[2]))

if __name__ == "__main__":
    solve()