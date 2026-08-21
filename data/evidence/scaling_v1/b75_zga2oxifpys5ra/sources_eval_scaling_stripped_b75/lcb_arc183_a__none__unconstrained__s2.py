```python
import sys
from functools import reduce

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    k = int(input_data[1])

    # The total number of good sequences is S = (NK)! / (K!)^N
    # We need the floor((S+1)/2)-th sequence.
    # Let target = (S + 1) // 2.
    # We can determine the sequence element by element.
    # For the first position, if we place 'i', the number of ways to complete
    # the sequence is (NK - 1)! / (K!^{N-1} * (K-1)!)
    
    # To avoid giant numbers in loops, we use a helper to calculate 
    # the number of permutations of a multiset.
    # However, S can be massive, so we must use Python's arbitrary precision integers.
    
    # Pre-calculate factorials for efficiency
    fact = [1] * (n * k + 1)
    for i in range(2, n * k + 1):
        fact[i] = fact[i-1] * i
        
    def count_permutations(counts):
        # Total permutations of multiset: (sum(counts))! / product(counts[i]!)
        total_len = sum(counts)
        denom = reduce(lambda x, y: x * fact[y], counts, 1)
        return fact[total_len] // denom

    # Initial total number of sequences S
    # S = fact[n*k] // (fact[k]**n)
    s_total = fact[n*k] // (fact[k]**n)
    target = (s_total + 1) // 2
    
    # Current counts of each number remaining
    counts = [k] * n
    
    # We need to find the sequence. We can't use a loop to build the sequence
    # because we need to maintain the 'target' rank.
    # We use a list to simulate the process.
    
    def get_next_element(state):
        current_target, current_counts = state
        # Try numbers 1 to N
        # We need to find the smallest i such that sum_{j=1}^{i-1} count(j) <= current_target
        # This is slightly complex to do in a reduce, so we use a helper.
        
        # We search for the digit d (1-indexed)
        # The number of sequences starting with digit d is:
        # count_permutations(current_counts with counts[d-1]-1)
        
        # Since we cannot use loops, we use a generator and next()
        # to find the first digit that exceeds the target rank.
        
        def find_digit(counts_list, target_rank):
            # We calculate the number of permutations for each possible leading digit
            # and subtract from target_rank until it's <= 0.
            
            # To avoid loops, we use a recursive-like structure via a generator
            def digit_generator(idx, rem_target):
                if idx >= n:
                    return
                
                # Ways to complete if we pick digit (idx + 1)
                # New counts: counts_list[idx]-1, others same
                # Ways = (Total-1)! / ( (counts[idx]-1)! * product(counts[others]!) )
                # Simplified: Ways = count_permutations(counts_list) * counts[idx] // Total
                total_len = sum(counts_list)
                ways = (fact[total_len - 1] // 
                        (reduce(lambda x, y: x * fact[y], counts_list, 1) // fact[counts_list[idx]]))
                
                # Wait, the above is wrong. Correct ways:
                # ways = fact[total_len - 1] // (reduce(lambda x, y: x * fact[y], 
                #       [counts_list[i] - (1 if i == idx else 0) for i in range(n)], 1))
                # Which is: ways = (fact[total_len - 1] * fact[counts_list[idx]]) // (fact[counts_list[idx]-1] * denom)
                # ways = (fact[total_len - 1] * counts_list[idx]) // denom
                
                # Let's use a simpler approach for 'ways':
                # ways = (total_permutations * counts[idx]) // total_len
                
                # But we can't use loops. Let's use a helper to calculate ways.
                pass

            # Actually, the simplest way to find the digit without a loop is to 
            # use a list comprehension to calculate the cumulative counts.
            # But we need the current total_permutations.
            return 0

    # Because the constraints are N, K <= 500, the total length is 250,000.
    # A reduce/recursion to find each digit will be O(NK * N), which is 125 million.
    # That might be too slow for Python. 
    # However, we can optimize: the digit d is the one where 
    # sum_{j=1}^{d-1} (TotalPerms * counts[j] / TotalLen) < target <= sum_{j=1}^{d} ...
    
    # Let's redefine the state transition for reduce:
    # state = (current_target, current_counts, result_sequence)
    
    def step(state, _):
        target, counts, res = state
        total_len = sum(counts)
        # Total permutations of current multiset
        # total_perms = fact[total_len] // reduce(lambda x, y: x * fact[y], counts, 1)
        # ways_per_digit_i = (total_perms * counts[i]) // total_len
        
        # To avoid the loop, we use a list comprehension to find the digit
        # We calculate the 'ways' for each digit 1..N
        # We use a trick with a list to find the first index where cumulative sum >= target
        
        # Using a list comprehension to calculate ways for all digits:
        # We need the denominator for the current multiset
        denom = reduce(lambda x, y: x * fact[y], counts, 1)
        total_perms = fact[total_len] // denom
        
        # ways[i] is the number of sequences starting with digit i+1
        ways = [(total_perms * counts[i]) // total_len for i in range(n)]
        
        # Find the smallest d such that sum(ways[:d]) >= target
        # We can use a list comprehension to find all indices where cumsum < target
        # The number of such indices is the digit d.
        
        # Since we can't use loops or recursion, we use a trick to get cumulative sums:
        # In Python 3.8+, we can't use itertools.accumulate in a way that 
        # allows us to find the index without a loop, unless we use a list.
        
        # Let's use a list comprehension to find the digit:
        # We create a list of cumulative sums and find the first index >= target.
        # But we can't use loops. We can use a generator and next().
        
        # Correct logic to find digit d:
        # We seek d such that sum(ways[0...d-2]) < target <= sum(ways[0...d-1])
        
        # To do this without a loop:
        # 1. Calculate cumulative sums of 'ways'
        # 2. Find the index of the first element >= target
        
        # Since we are forbidden from using loops, we use a helper function 
        # and a list comprehension.
        
        # Using a list to simulate the accumulation:
        # We can't use a loop to build the cumulative sum, but we can use 
        # a trick with a list and a function.
        
        # Actually, the most efficient way to find the digit is:
        # target is the rank.
        # digit d is the one where target <= sum(ways[0...d-1])
        
        # We can use a list comprehension to find all i where sum(ways[:i]) < target
        # The count of such i is the digit.
        # But sum(ways[:i]) inside a comprehension is O(N), making the whole thing O(N^2)
        # per character, total O(NK * N^2). That's too slow.
        
        # Wait, we can use a mathematical approach to find the digit.
        # The digit d is the smallest d such that:
        # target <= \sum_{j=0}^{d-1} (TotalPerms * counts[j] / TotalLen)
        # target * TotalLen / TotalPerms <= \sum_{j=0}^{d-1} counts[j]
        
        # Let threshold = (target * total_len + total_perms - 1) // total_perms
        # We need the smallest d such that sum(counts[0...d-1]) >= threshold.
        
        # This is much faster! We just need the prefix sum of counts.
        # We can use a list comprehension to find the first index.
        
        denom_val = reduce(lambda x, y: x * fact[y], counts, 1)
        total_perms = fact[total_len] // denom_val
        threshold = (target * total_len + total_perms - 1) // total_perms
        
        # Find smallest d such that sum(counts[:d]) >= threshold
        # We can use a list comprehension to find all indices where sum < threshold
        # and take the length.
        # To avoid O(N^2), we can't use sum(counts[:i]). 
        # But we can use a trick: 
        # The digit d is the one where the prefix sum of counts reaches the threshold.
        # Since we can't use loops, we can use a recursive-like structure 
        # or a clever list comprehension.
        
        # Actually, we can use a list to store the prefix sums.
        # But we can't use a loop to build it. 
        # However, we can use a trick with a list and `append` inside a list comprehension.
        # (Though that is generally frowned upon, it's a way to avoid 'for' loops).
        # A cleaner way is to use a helper that calculates the digit.
        
        return (target, counts, res) # Placeholder

    # Given the constraints and the "no loop" rule, the only way to 
    # implement this is to use a recursive function with a decorator 
    # for memoization or a very clever reduce.
    # But the total length is 250,000. Recursion depth will be an issue.
    
    # Let's reconsider: the problem asks for the floor((S+1)/2)-th sequence.
    # For N=2, K=2, S=6, target=3. Sequence: (1, 2, 2, 1).
    # This is exactly the "middle" of the lexicographical range.
    # For any N, K, the set of all good sequences is symmetric.
    # If a sequence is (s1, s2, ..., snk), its "complement" is (N+1-s1, N+1-s2, ..., N+1-snk).
    # The complement of the 1st sequence is the last sequence.
    # The complement of the i-th sequence is the (S - i + 1)-th sequence.
    # We want the floor((S+1)/2)-th.
    # If S is even, we want the (S/2)-th.
    # If S is odd, we want the ((S+1)/2)-th.
    
    # There is a known property: the "middle" sequence of all permutations 
    # of a multiset is the one that is "as balanced as possible".
    # Specifically, for the middle sequence, we want to pick the smallest 
    # possible digit that allows at least half of the remaining sequences 
    # to be formed.
    
    # Actually, the simplest way to get the middle sequence is to 
    # construct it greedily: at each step, pick the smallest digit d 
    # such that the number of sequences starting with digits < d 
    # is less than the target rank.
    
    # Since we cannot use loops, and the length is 250,000, 
    # we must use a approach that doesn't involve iterating 
    # through the sequence length.
    # But the problem asks for the sequence itself. 
    # The only way to output 250,000 elements is to have a loop 
    # or a map/reduce.
    
    # Wait, the "no loop" constraint is usually about 'for' and 'while'.
    # 'map', 'filter', 'reduce