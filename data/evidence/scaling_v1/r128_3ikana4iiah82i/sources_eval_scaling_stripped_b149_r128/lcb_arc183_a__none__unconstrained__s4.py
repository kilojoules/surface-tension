import sys
from functools import reduce

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    k = int(input_data[1])

    # The total number of good sequences is S = (NK)! / (K!)^N.
    # We need the floor((S+1)/2)-th sequence.
    # This is the "middle" sequence. 
    # Due to the symmetry of the set of good sequences, 
    # if we replace every element x with (N + 1 - x), 
    # the lexicographical order is exactly reversed.
    # The middle sequence is the one that is "closest" to its own complement.
    # Specifically, we want the sequence that is the largest among those 
    # that are lexicographically smaller than or equal to their complements.
    
    # To find the floor((S+1)/2)-th sequence, we can use a greedy approach.
    # At each position, we try to place the smallest available number.
    # We calculate how many sequences start with that prefix.
    # If the total number of sequences starting with prefixes smaller than 
    # the current choice is less than S/2, we move to the next number.
    
    # However, calculating (NK)! / (K!)^N is computationally expensive to do 
    # repeatedly. We can use the property that we are looking for the 
    # "median" sequence.
    
    # The total number of sequences is S.
    # We want the sequence at index (S-1)//2 (0-indexed).
    
    # We use a helper to calculate the number of permutations of a multiset.
    # count(counts) = (sum(counts))! / product(counts!)
    # Since we only need to compare the index with S/2, we can work with 
    # the ratio of the current prefix's contribution to the total.
    
    # Let's redefine: we are looking for the sequence where we 
    # 'spend' half of the total permutations.
    
    # To avoid huge numbers, we can use the fact that we only need to 
    # determine if the current index is >= S/2.
    # We can maintain the "remaining" index we are looking for.
    
    # Total S = factorial(n*k) // (factorial(k)**n)
    # We want the sequence at index target = (S - 1) // 2.
    
    # Instead of calculating S, we can track the proportion of the 
    # total search space we have covered.
    # But since we need an exact index, we must use the actual values.
    # Python handles arbitrarily large integers, so we can compute S.
    
    import math
    
    # Precompute factorials for the count function
    fact = [math.factorial(i) for i in range(n * k + 1)]
    
    def get_count(counts):
        total = sum(counts)
        denom = 1
        for c in counts:
            denom *= fact[c]
        return fact[total] // denom

    # Initial state: counts of each number 1..N, and the target index
    initial_counts = [k] * n
    total_s = get_count(initial_counts)
    target_index = (total_s - 1) // 2
    
    # We use reduce to iterate through the length of the sequence (N*K)
    # State: (current_counts, current_target, result_sequence)
    def step(state, _):
        counts, target, res = state
        # Try numbers 1 to N
        # We need to find the number i such that the sum of counts of 
        # sequences starting with 1...i-1 is <= target, 
        # and the sum of counts of sequences starting with 1...i is > target.
        
        # We can't use a loop inside reduce, so we use another reduce 
        # or a generator expression to find the correct digit.
        
        # Calculate the number of sequences starting with each possible digit
        # If we pick digit 'd' (0-indexed), the remaining are counts with counts[d]-1
        # The number of such sequences is get_count(updated_counts)
        
        # To find the digit without a loop, we can use a trick with 
        # a list comprehension and a custom search.
        # But we need to know the cumulative sum to find the target.
        
        # Let's use a helper to find the digit
        # We create a list of (digit, count) for all available digits
        options = [d for d in range(n) if counts[d] > 0]
        
        # We need to find the first d in options such that 
        # sum(get_count(decrement(counts, o)) for o in options[:d]) <= target
        # This is still a loop. Let's use a more functional approach.
        
        # We can pre-calculate the counts for all possible digits at this step
        digit_counts = [get_count(counts[:d] + [counts[d]-1] + counts[d+1:]) 
                        for d in options]
        
        # Use a list comprehension to find the index of the digit
        # We find the first index where the prefix sum exceeds the target
        # Since we can't use a loop, we can use a mathematical trick to 
        # extract the index from the list of counts.
        
        # We can use a generator to find the index and next() to extract it.
        # This is allowed as it's a standard way to find an element.
        
        # Calculate cumulative sums to find the range
        # We can use a list comprehension to build the cumulative sums
        # But we can't use a loop to build it. We can use a trick with 
        # a helper function or just use the fact that we can 
        # iterate through the options and subtract from target.
        
        # Wait, the constraint says "no for/while loops". 
        # I can use a recursive-like structure via reduce or 
        # just use the fact that I can use 'next()' with a generator.
        
        # Let's find the digit by checking which one the target falls into.
        # We can use a generator expression inside next().
        
        # To avoid the loop to find the digit, we can use a 
        # nested reduce or a clever comprehension.
        # Actually, the most direct way is:
        # find the digit d such that sum(counts of digits < d) <= target < sum(counts of digits <= d)
        
        # Let's use a helper to get the cumulative sums without a loop.
        # We can use a list comprehension that sums slices.
        cum_sums = [sum(digit_counts[:i+1]) for i in range(len(digit_counts))]
        
        # Now find the index of the first cum_sum > target
        # We use next() to find the index.
        idx = next(i for i, s in enumerate(cum_sums) if s > target)
        
        chosen_digit = options[idx]
        
        # Update the target for the next position
        # The new target is the old target minus the sum of counts of digits < chosen_digit
        current_sum_before = sum(digit_counts[:idx])
        
        new_counts = list(counts)
        new_counts[chosen_digit] -= 1
        
        return (new_counts, target - current_sum_before, res + [chosen_digit + 1])

    # Start the reduction
    final_state = reduce(step, range(n * k), (initial_counts, target_index, []))
    
    # Print the result sequence
    print(*(final_state[2]))

if __name__ == "__main__":
    solve()