import sys
from math import comb
from functools import reduce

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    K = int(input_data[1])

    # The total number of good sequences S is (NK)! / (K!)^N
    # We need the floor((S+1)/2)-th sequence.
    # Let target_rank = (S + 1) // 2.
    # Since S can be very large, we handle the rank dynamically.
    
    # Function to calculate total permutations of a multiset
    # Total = (sum(counts))! / product(counts[i]!)
    # However, we only need to compare the target rank with the number of 
    # sequences starting with a certain digit.
    
    # Let's use a helper to calculate permutations of remaining elements.
    # Permutations = (sum(counts))! / product(counts[i]!)
    # To avoid huge factorials, we can use a property:
    # If we fix the first element as 'i', the number of ways to arrange the rest is:
    # (Total_Remaining - 1)! / (K1! * ... * (Ki-1)! * ... * KN!)
    # where Total_Remaining is the current length of the sequence to be filled.
    
    # Instead of calculating S first, we can maintain the target rank.
    # The total number of sequences S is:
    # S = math.comb(N*K, K) * math.comb((N-1)*K, K) * ... * math.comb(K, K)
    
    # We need rank = (S + 1) // 2.
    # Because S is symmetric (if a sequence is X, its "complement" is N+1-X),
    # the middle of the lexicographical order is reached when we've passed 
    # half of the total permutations.
    
    # Let's pre-calculate the total S to find the target rank.
    # Since we need (S+1)//2, we can use a large integer.
    def get_total_permutations(counts):
        total_len = sum(counts)
        res = 1
        current_len = total_len
        for c in counts:
            res *= comb(current_len, c)
            current_len -= c
        return res

    total_s = get_total_permutations([K] * N)
    target_rank = (total_s + 1) // 2

    # We need to find the sequence. We determine it element by element.
    # state: (current_counts, current_rank)
    # We use reduce to simulate the loop over the length of the sequence (N*K).
    
    def get_next_element(state, _):
        counts, rank = state
        # Try digits 1 to N
        # For digit d, the number of sequences starting with d is:
        # (Total_Remaining - 1)! / (counts[0]! * ... * (counts[d-1]-1)! * ... * counts[N-1]!)
        # This is equal to: Total_Permutations(counts) * counts[d-1] / Total_Remaining
        
        total_rem = sum(counts)
        total_perms = get_total_permutations(counts)
        
        # We need to find d such that the sum of perms for 1..d-1 < rank <= sum of perms for 1..d
        # We can use a generator and a custom reduce or a loop-like structure to find d.
        # Since we can't use loops, we use a recursive-like search or a list comprehension.
        
        def find_digit(d, accumulated_rank):
            if d > N: return None # Should not happen
            # Number of permutations if we pick digit d
            # ways = total_perms * counts[d-1] // total_rem
            if counts[d-1] == 0:
                return find_digit(d + 1, accumulated_rank)
            
            ways = (total_perms * counts[d-1]) // total_rem
            if accumulated_rank <= ways:
                return d, accumulated_rank
            return find_digit(d + 1, accumulated_rank - ways)

        digit, new_rank = find_digit(1, rank)
        
        # Update counts
        new_counts = list(counts)
        new_counts[digit-1] -= 1
        
        return (tuple(new_counts), new_rank), digit

    # Initial state: (counts, target_rank)
    initial_state = (tuple([K] * N), target_rank)
    
    # Use reduce to build the sequence
    # The accumulator stores (current_state, sequence_list)
    final_result = reduce(
        lambda acc, _: (get_next_element(acc[0], _), acc[1] + [get_next_element(acc[0], _)[1]], 
                        # Wait, the above logic is slightly flawed because get_next_element 
                        # is called twice. Let's fix it.
                        None), 
        range(N * K), 
        (initial_state, [])
    )
    
    # Redefining the reduce logic to avoid double calling and maintain state correctly.
    def step(acc, _):
        state, seq = acc
        (new_state, digit) = get_next_element(state, _)
        return (new_state, seq + [digit])

    final_output = reduce(step, range(N * K), (initial_state, []))
    print(*(final_output[1]))

# To avoid the double-call and recursion depth issues, 
# I will rewrite the logic slightly to fit the constraints perfectly.
def solve_final():
    input_data = sys.stdin.read().split()
    if not input_data: return
    N, K = map(int, input_data)
    
    def get_total_perms(counts):
        res, curr = 1, sum(counts)
        # Using a list comprehension and reduce to multiply combinations
        return reduce(lambda a, b: a * b, [comb(curr - sum(counts[:i]), counts[i]) for i in range(N)], 1)

    total_s = get_total_perms([K] * N)
    target_rank = (total_s + 1) // 2

    def get_digit_and_rank(counts, rank):
        total_rem = sum(counts)
        total_perms = get_total_perms(counts)
        
        # We need to find the digit d. We can use a list comprehension to calculate 
        # the number of permutations for each digit and then find the index.
        # ways_per_digit = [ (total_perms * counts[i]) // total_rem if counts[i] > 0 else 0 for i in range(N)]
        # But we can't use loops. We use map.
        ways = list(map(lambda c: (total_perms * c) // total_rem if c > 0 else 0, counts))
        
        # Find the first index where the prefix sum of ways >= rank
        # We can use a scan (prefix sum) via a trick with reduce
        def find_idx(r, w_list):
            # Use reduce to find the index and the remaining rank
            # acc = (current_rank, found_index)
            res = reduce(lambda a, w: (a[0] - w, a[1] + 1) if a[0] > w and a[1] == -1 else 
                         (a[0] - w if a[1] == -1 else a[0], a[1]) 
                         for w in w_list) # This is getting complex.
            # Let's use a simpler approach for finding the index.
            pass

    # Actually, the most reliable way to find the digit without loops/recursion 
    # is to use a generator expression and next().
    def get_next(state, _):
        counts, rank = state
        total_rem = sum(counts)
        total_perms = get_total_perms(counts)
        
        # Generator to find the digit
        def gen_digit(r, c_idx):
            if c_idx >= N: return None
            c = counts[c_idx]
            ways = (total_perms * c) // total_rem if c > 0 else 0
            if c > 0 and r <= ways:
                return (c_idx + 1, r)
            return gen_digit(r - ways, c_idx + 1)
        
        # Since recursion is allowed but loops aren't, gen_digit is fine.
        # However, for N=500, recursion depth might be an issue. 
        # But we only recurse N times. sys.setrecursionlimit is needed.
        res = gen_digit(rank, 0)
        digit, new_rank = res
        new_counts = list(counts)
        new_counts[digit-1] -= 1
        return (tuple(new_counts), new_rank), digit

# Given the constraints and the "no loop" rule, 
# the most idiomatic way is using reduce and a helper function.
# I will implement the final version now.

import sys
from math import comb
from functools import reduce

# Increase recursion depth for the digit search
sys.setrecursionlimit(2000)

def solve_real():
    input_data = sys.stdin.read().split()
    if not input_data: return
    N, K = map(int, input_data)
    
    def get_total_perms(counts):
        # Total permutations of multiset: (sum(counts))! / product(counts[i]!)
        # Calculated as product of comb(remaining, count)
        return reduce(lambda a, b: a * b, 
                      [comb(sum(counts[i:]), counts[i]) for i in range(N)], 1)

    total_s = get_total_perms([K] * N)
    target_rank = (total_s + 1) // 2

    def find_digit(rank, counts, total_perms, total_rem, idx):
        if idx >= N: return None
        c = counts[idx]
        ways = (total_perms * c) // total_rem if c > 0 else 0
        if c > 0 and rank <= ways:
            return (idx + 1, rank)
        return find_digit(rank - ways, counts, total_perms, total_rem, idx + 1)

    def step(acc, _):
        (counts, rank), seq = acc
        total_rem = sum(counts)
        total_perms = get_total_perms(counts)
        digit, new_rank = find_digit(rank, counts, total_perms, total_rem, 0)
        new_counts = list(counts)
        new_counts[digit-1] -= 1
        return ((tuple(new_counts), new_rank), seq + [digit])

    initial_state = ((tuple([K] * N), target_rank), [])
    final_result = reduce(step, range(N * K), initial_state)
    print(*(final_result[1]))

if __name__ == "__main__":
    solve_real()