import sys
from functools import reduce
from math import factorial

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    K = int(input_data[1])

    # The total number of good sequences S is (N*K)! / (K!)^N
    # We need the floor((S+1)/2)-th sequence.
    # Let's calculate S.
    # Since S can be very large, we use Python's arbitrary precision integers.
    
    # Precompute factorials for the multinomial coefficient
    # The number of ways to arrange items with counts c1, c2, ..., cN is
    # (sum(ci))! / (c1! * c2! * ... * cN!)
    
    # We need a way to calculate the number of permutations given current counts
    # Using a helper function inside reduce.
    
    def get_count(counts):
        total = sum(counts)
        # Multinomial coefficient: total! / product(c!)
        # We can use a more efficient way, but N, K <= 500 allows for direct calculation
        # However, we only need to compare the target index with the count of sequences
        # starting with 1, then 2, etc.
        # The number of sequences starting with digit i is:
        # (total - 1)! / (c1! ... (ci-1)! ... (cN-1)! ... cN!)
        # Which is: [total! / (c1! ... cN!)] * (ci / total)
        pass

    # To avoid loops, we use reduce to build the sequence.
    # State: (current_counts, target_index, result_sequence)
    # target_index is 1-based.
    
    # Initial S calculation
    # S = factorial(N*K) // (factorial(K)**N)
    # target = (S + 1) // 2
    
    # To avoid recalculating large factorials, we observe that the number of sequences
    # starting with digit i is: (Total-1)! / ( (K1)!...(Ki-1)!...(KN)! )
    # where Ki is the remaining count of digit i.
    
    # Let f(counts) = (sum(counts))! / product(counts!)
    # The number of sequences starting with digit i is f(counts) * counts[i] / sum(counts)
    
    # Since we cannot use loops, we use a list comprehension to find the digit
    # and reduce to iterate through the length of the sequence.
    
    def get_s(n, k):
        # Using a property: S = (N*K)! / (K!)^N
        # But we can just compute it once.
        res = factorial(n * k)
        denom = factorial(k)**n
        return res // denom

    total_s = get_s(N, K)
    target = (total_s + 1) // 2
    
    # State: (counts, current_target, sequence)
    # counts: list of remaining counts for each digit 1..N
    initial_counts = [K] * N
    
    def step(state, _):
        counts, t, seq = state
        total_rem = sum(counts)
        
        # We need to find the smallest digit d such that 
        # sum_{j=1}^{d-1} (ways starting with j) < t <= sum_{j=1}^{d} (ways starting with j)
        
        # ways_starting_with_j = (total_rem - 1)! / (c1! ... (cj-1)! ... cN!)
        # = [ (total_rem)! / (c1! ... cN!) ] * cj / total_rem
        
        # Let current_total_ways = (total_rem)! / product(c!)
        # ways_j = current_total_ 그리고 (counts[j-1] / total_rem)
        
        # To avoid floating point and repeated factorial calls:
        # current_total_ways = factorial(total_rem)
        # for c in counts: current_total_ways //= factorial(c)
        
        # But we can't use a loop to find d. We use a list comprehension and next().
        # We pre-calculate the ways for each digit.
        
        # current_total_ways is the number of permutations of the remaining elements.
        # The number of permutations starting with digit j is:
        # (total_rem - 1)! / (c1! ... (cj-1)! ... cN!)
        
        # Let's compute the common part: (total_rem - 1)! / product(c!)
        # Then ways_j = common_part * cj
        
        # To compute common_part:
        # common_part = factorial(total_rem - 1)
        # for c in counts: common_part //= factorial(c)
        # Wait, the denominator is c1! * c2! ... cN!. 
        # But for digit j, the denominator is c1! ... (cj-1)! ... cN!.
        # So ways_j = (total_rem - 1)! / (c1! ... (cj-1)! ... cN!)
        # ways_j = [ (total_rem - 1)! / (c1! ... cN!) ] * cj
        
        # Let's use a simpler approach:
        # The number of ways to arrange the remaining is W = total_rem! / product(c!)
        # The number of ways starting with digit j is W * cj / total_rem.
        
        # Since we need to avoid loops, we use a helper to calculate W.
        # We can't define a function with a loop inside, but we can use reduce.
        
        denom = reduce(lambda x, y: x * factorial(y), counts, 1)
        W = factorial(total_rem) // denom
        
        # Find digit d (1-indexed)
        # We use a list comprehension to find the first d where the prefix sum of ways >= t.
        # ways_j = (W * counts[j-1]) // total_rem
        
        # To find d without a loop, we can use a list of cumulative sums.
        # But we can't use a loop to build the cumulative sum.
        # We can use a list comprehension with a slice and sum().
        
        # d is the smallest index such that sum( (W * counts[j]) // total_rem for j in range(d) ) >= t
        # Since N is small (500), we can just iterate d from 0 to N-1.
        
        # We use a generator expression inside next() to find the digit.
        d = next(j for j in range(N) if (
            sum((W * counts[m]) // total_rem for m in range(j)) < t <= 
            sum((W * counts[m]) // total_rem for m in range(j + 1))
            and counts[j] > 0
        ))
        
        # Update target for the next position
        # new_t = t - sum(ways for j in range(d))
        new_t = t - sum((W * counts[m]) // total_rem for m in range(d))
        
        # Update counts
        new_counts = [counts[i] - (1 if i == d else 0) for i in range(N)]
        
        return (new_counts, new_t, seq + [d + 1])

    # Use reduce to simulate the process for NK steps
    final_state = reduce(step, range(N * K), (initial_counts, target, []))
    
    # Print the result sequence
    print(*(final_state[2]))

if __name__ == "__main__":
    solve()