import sys
from math import comb
from functools import reduce

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    k = int(input_data[1])

    # The total number of good sequences is the multinomial coefficient:
    # S = (N*K)! / (K!)^N
    # However, we need the floor((S+1)/2)-th sequence.
    # We can determine the sequence digit by digit.
    
    # Function to calculate the number of ways to complete a sequence
    # given the remaining counts of each number.
    def count_ways(counts):
        # Total remaining slots
        total = sum(counts)
        # Multinomial coefficient: total! / (c1! * c2! ... * cn!)
        # We can compute this using combinations:
        # comb(total, c1) * comb(total-c1, c2) * ...
        return reduce(lambda acc, c: acc * comb(total - (sum(counts[:counts.index(c)]) if counts.index(c) > 0 else 0), c), 
                      counts, 1)
    
    # Since the above reduce is slightly wrong due to index(), 
    # let's use a more robust way to calculate multinomials.
    def get_multinomial(counts):
        res = 1
        rem = sum(counts)
        # We simulate the product of comb(rem, c) for c in counts
        # Using a helper to track the remaining total
        def step(state, c):
            curr_rem, curr_res = state
            return (curr_rem - c, curr_res * comb(curr_rem, c))
        
        return step(reduce(step, counts, (rem, 1)))[1]

    # The target index is floor((S + 1) / 2)
    # S = get_multinomial([k] * n)
    total_s = get_multinomial([k] * n)
    target = (total_s + 1) // 2

    # We need to find the sequence. We'll use reduce to iterate through the length NK.
    # State: (current_counts, current_target, result_sequence)
    def find_next_digit(state, _):
        counts, target_idx, seq = state
        
        # Try digits 1 to N
        def search_digit(digit, current_target):
            if counts[digit-1] > 0:
                # Calculate ways if we pick this digit
                # Create a temporary count list
                temp_counts = list(counts)
                temp_counts[digit-1] -= 1
                ways = get_multinomial(temp_counts)
                
                if current_target <= ways:
                    return (digit, current_target)
                else:
                    # If we haven't reached the target, subtract ways and try next digit
                    # But search_digit needs to be called in a loop-like fashion.
                    # Since we can't use loops, we'll handle this via a recursive-like 
                    # structure inside a reduce or a custom function.
                    return None
            return None

        # To avoid loops, we use a reduce to find the correct digit and update target
        def pick_digit(acc, d):
            curr_counts, curr_target, found_digit = acc
            if found_digit is not None:
                return acc
            
            temp_counts = list(curr_counts)
            temp_counts[d-1] -= 1
            ways = get_multinomial(temp_counts)
            
            if curr_target <= ways:
                return (curr_counts, curr_target, d)
            else:
                return (curr_counts, curr_target - ways, None)

        # We need to handle the counts update carefully.
        # The 'acc' in pick_digit is (counts, target, found_digit)
        # But we need to check if counts[d-1] > 0 first.
        def pick_digit_safe(acc, d):
            curr_counts, curr_target, found_digit = acc
            if found_digit is not None or curr_counts[d-1] == 0:
                return acc
            
            temp_counts = list(curr_counts)
            temp_counts[d-1] -= 1
            ways = get_multinomial(temp_counts)
            
            if curr_target <= ways:
                return (curr_counts, curr_target, d)
            else:
                return (curr_counts, curr_target - ways, None)

        final_state = reduce(pick_digit_safe, range(1, n + 1), (counts, target_idx, None))
        
        # Update counts and sequence for the next position
        new_counts = list(counts)
        digit = final_state[2]
        new_counts[digit-1] -= 1
        return (new_counts, final_state[1], seq + [digit])

    # Initial state: (counts, target, sequence)
    initial_state = ([k] * n, target, [])
    final_result = reduce(find_next_digit, range(n * k), initial_state)
    
    print(*(final_result[2]))

if __name__ == "__main__":
    solve()