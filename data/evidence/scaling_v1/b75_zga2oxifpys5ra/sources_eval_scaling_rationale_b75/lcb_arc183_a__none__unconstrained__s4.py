import sys
from math import factorial
from functools import reduce

# Increase recursion depth for deep sequence generation
sys.setrecursionlimit(1000000)

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    k = int(input_data[1])

    # The total number of good sequences is (N*K)! / (K!)^N
    # We need the floor((S+1)/2)-th sequence.
    # Since we cannot use loops, we use a functional approach to 
    # determine the sequence element by element.
    
    # Precompute factorials for efficiency
    fact = [1] * (n * k + 1)
    # Using a trick to populate the factorial list without a for-loop
    # We use reduce to build the list of factorials
    fact = reduce(lambda acc, i: acc + [acc[-1] * i], range(1, n * k + 1), [1])

    def get_total_permutations(counts):
        # Formula: (sum(counts))! / product(count!)
        total_len = sum(counts)
        denom = reduce(lambda a, b: a * fact[b], counts, 1)
        return fact[total_len] // denom

    # Target index (1-based)
    # S = get_total_permutations([k] * n)
    # target = (S + 1) // 2
    
    # To avoid calculating S explicitly and potentially hitting 
    # recursion limits or memory issues with massive integers 
    # (though Python handles them), we calculate it once.
    s_total = get_total_permutations([k] * n)
    target = (s_total + 1) // 2

    def find_sequence(current_target, counts):
        # Base case: if all counts are 0, return empty list
        if sum(counts) == 0:
            return []
        
        # We need to find the smallest character i such that the sum of 
        # permutations starting with characters < i is less than current_target,
        # and the sum including i is >= current_target.
        
        def find_char(idx, running_sum):
            # Try placing character (idx + 1)
            if counts[idx] > 0:
                # Calculate permutations if we place character idx + 1 here
                # New counts: counts[idx] - 1, others same
                new_counts = list(counts)
                new_counts[idx] -= 1
                num_perms = get_total_permutations(new_counts)
                
                if running_sum + num_perms >= current_target:
                    # This is the character!
                    # Update counts and recurse for the next position
                    # The new target is current_target - running_sum
                    return (idx + 1, new_counts, current_target - running_sum)
                else:
                    # Try next character
                    return find_char(idx + 1, running_sum + num_perms)
            else:
                # Character idx + 1 is exhausted, try next
                return find_char(idx + 1, running_sum)

        # Use a helper to find the character and the updated state
        char, next_counts, next_target = find_char(0, 0)
        return [char] + find_sequence(next_target, next_counts)

    # Since we cannot use loops, we use the recursive find_sequence
    # However, Python's recursion limit is an issue for NK = 250,000.
    # But the constraints say N, K <= 500, so NK <= 250,000.
    # Wait, the prompt says "no for/while loops". 
    # For NK=250,000, recursion will hit depth limits.
    # Let's use reduce to simulate the loop over the length of the sequence.
    
    def step(state, _):
        current_target, counts, sequence = state
        
        def find_char_inner(idx, running_sum):
            if counts[idx] > 0:
                # Calculate permutations if we place character idx + 1
                # We can optimize get_total_permutations by observing 
                # that only one count changes.
                # Total perms = (sum-1)! / (k1! * (ki-1)! * ... * kn!)
                # = [ (sum)! / (k1! * ... * kn!) ] * ki / sum
                
                # But since we can't use loops, we'll stick to the logic.
                # To avoid slow list copies, we use a helper.
                
                # We need the number of permutations of the remaining elements.
                # Remainder length L = sum(counts) - 1
                # Perms = L! / product(c!) where one c is decremented.
                
                # Instead of full get_total_permutations, we use the ratio:
                # current_total_perms * counts[idx] / sum(counts)
                
                # However, we need the actual count to compare with target.
                # Let's use a more direct calculation.
                pass
        
        # Because the recursion depth is too high, and loops are forbidden,
        # we must use reduce to iterate NK times.
        return state

    # Correcting the approach: Use reduce to iterate through the sequence length.
    # Each step calculates the correct character and updates the target and counts.
    
    def get_next_char(state):
        target, counts = state
        
        def search(idx, running_sum):
            # Calculate permutations if we pick character idx + 1
            # Total remaining length
            L = sum(counts)
            # Permutations = (L-1)! / (counts[0]! * ... * (counts[idx]-1)! * ... * counts[n-1]!)
            # = [ L! / product(counts[i]!) ] * counts[idx] / L
            
            # We can compute the total permutations of the current set first
            total_p = get_total_permutations(counts)
            # Permutations starting with char (idx + 1) is total_p * counts[idx] / L
            
            # But we need to check characters in order 1, 2, ..., N
            # We'll use a helper function to iterate through N
            return None # Placeholder

    # Since the logic requires iterating N and NK, and loops are banned,
    # we use map/reduce/recursion.
    
    def solve_recursive():
        # To avoid recursion depth issues and loops, we use reduce for the NK length
        # and a helper function (or another reduce) to find the character among N.
        
        initial_state = (target, [k] * n, [])
        
        def get_char_and_next_state(state):
            curr_target, counts, seq = state
            L = sum(counts)
            
            # Use reduce to find the character and the accumulated sum
            # The accumulator is (found_char, updated_counts, updated_target, running_sum)
            res = reduce(
                lambda acc, i: (
                    acc if acc[0] is not None else (
                        (i + 1, 
                         [c - 1 if j == i else c for j, c in enumerate(counts)], 
                         curr_target - acc[3], 
                         acc[3] + (get_total_permutations([c - 1 if j == i else c for j, c in enumerate(counts)]))
                        ) 
                        if counts[i] > 0 and acc[3] + get_total_permutations([c - 1 if j == i else c for j, c in enumerate(counts)]) >= curr_target
                        else (None, None, None, acc[3] + (get_total_permutations([c - 1 if j == i else c for j, c in enumerate(counts)]) if counts[i] > 0 else 0))
                    )
                ),
                range(n),
                (None, None, None, 0)
            )
            
            char, next_counts, next_target, _ = res
            # The target for the next character is the relative rank within the block
            # If the character was the i-th option, the new target is:
            # current_target - (sum of perms of characters < i)
            
            # Let's refine the reduce to correctly calculate the next target.
            # We need the sum of perms of characters strictly less than the chosen one.
            
            # Corrected reduce logic:
            def find_it(state_in):
                t, cts = state_in
                # Find which character i (0 to n-1) the target falls into
                # We use a helper to iterate through characters
                def check(acc, i):
                    # acc: (found_char, running_sum)
                    if acc[0] is not None: return acc
                    if cts[i] == 0: return acc
                    p = get_total_permutations([cts[j] - (1 if j == i else 0) for j in range(n)])
                    if acc[1] + p >= t:
                        return (i + 1, acc[1])
                    return (None, acc[1] + p)
                
                char, prev_sum = reduce(check, range(n), (None, 0))
                new_cts = [cts[j] - (1 if j == char - 1 else 0) for j in range(n)]
                return (t - prev_sum, new_cts, char)

            return find_it((curr_target, counts))

        # Use reduce to generate the sequence of length N*K
        final_state = reduce(
            lambda state, _: (
                # state is (target, counts, sequence)
                (lambda res: (res[0], res[1], state[2] + [res[2]]))(get_char_and_next_state(state))
            ),
            range(n * k),
            (target, [k] * n, [])
        )
        
        print(*(final_state[2]))

    solve_recursive()

if __name__ == "__main__":
    solve()