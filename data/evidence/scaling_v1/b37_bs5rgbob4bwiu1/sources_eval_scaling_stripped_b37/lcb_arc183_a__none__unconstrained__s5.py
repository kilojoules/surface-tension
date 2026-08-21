```python
import sys

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    K = int(input_data[1])

    # The total number of good sequences S is (NK)! / (K!)^N.
    # We need the floor((S+1)/2)-th sequence.
    # This is exactly the middle sequence if S is odd, 
    # or the (S/2)-th sequence if S is even.
    # In both cases, it is the last sequence of the first half of the lexicographical order.
    # A key property of lexicographical order for permutations of a multiset is that
    # the sequence S_rev (where each element x is replaced by N - x + 1) 
    # and the sequence reversed in order are related.
    # Specifically, the "complement" of a sequence (replacing i with N-i+1)
    # maps the k-th lexicographical sequence to the (S - k + 1)-th sequence.
    # The middle index is (S+1)//2.
    # If we cannot compute S (which is huge), we can use the property that
    # the sequence at index (S+1)//2 is the one that is "self-complementary" 
    # in a sense, or specifically, it's the sequence that, when you replace 
    # each element x with (N - x + 1), the resulting sequence is the 
    # lexicographical "opposite".
    
    # For a given N and K, the sequence at index (S+1)//2 is constructed by:
    # For each position i from 1 to NK:
    # We try placing the smallest available number v (1 to N).
    # We check if the number of sequences starting with the current prefix 
    # is enough to reach the target index.
    # However, we can't compute the number of sequences.
    # Instead, we use the symmetry: the sequence at (S+1)//2 is the one 
    # where we try to keep the sequence "balanced".
    # The target index is exactly the middle. This means for every choice,
    # we want to know if the number of permutations starting with 1, 2... v-1
    # is less than S/2.
    
    # The number of ways to arrange the remaining elements is:
    # (sum(rem))! / product(rem_i!)
    # We compare this count to S/2.
    # Since we only need to know if the count is < S/2, and S/2 is 
    # (NK)! / (2 * (K!)^N), we are comparing:
    # (Total remaining)! / product(rem_i!)  vs  (NK)! / (2 * (K!)^N)
    
    # This is still hard. Let's use the property:
    # The sequence at index (S+1)//2 is the one that is "lexicographically"
    # the middle. For N=2, K=2, S=6, index=3. Sequences: 1122, 1212, 1221... Ans: 1221.
    # Note that 1221 is the "reverse complement" of itself? 
    # Complement of 1221 is 2112. Reverse of 2112 is 2112. No.
    # Actually, the sequence at (S+1)//2 is the one that is the 
    # lexicographical mirror of the sequence at (S - (S+1)//2 + 1).
    # The mirror of a sequence (a_1, ..., a_m) is (N-a_1+1, ..., N-a_m+1).
    # The sequence at index k and the sequence at index S-k+1 are mirrors.
    # So for k = (S+1)//2, the sequence is "almost" its own mirror.
    
    # Correct logic for "middle" sequence of a multiset permutation:
    # We can use a greedy approach with a custom comparison function 
    # or use the fact that we can use logarithms to compare large factorials.
    # However, there is a simpler way: the middle sequence is the one 
    # where we pick the smallest v such that the number of permutations 
    # starting with prefixes smaller than v is < S/2.
    
    # To compare (rem_total)! / product(rem_i!)  with  S/2:
    # We can use math.lgamma for log-factorials.
    import math
    
    def get_log_count(counts):
        total = sum(counts)
        return math_lgamma(total) - sum(math_lgamma(c) for c in counts)

    math_lgamma = math.lgamma

    # Target log value: log(S/2) = log(S) - log(2)
    # log(S) = lgamma(N*K + 1) - N * lgamma(K + 1)
    log_S_half = math_lgamma(N * K + 1) - N * math_lgamma(K + 1) - math.log(2)
    
    # We need the (S+1)//2 -th sequence.
    # Let target_rank = (S + 1) // 2.
    # Since we can't handle target_rank directly, we maintain it as a log 
    # or use the property that we only need to compare the current 
    # rank offset with the total remaining permutations.
    
    # Actually, the most robust way to find the middle element without 
    # huge numbers is to use the fact that we are looking for the 
    # sequence that splits the set in half.
    # We can maintain the "rank" we are looking for. 
    # Since we start at (S+1)//2, we can use a floating point 
    # approximation for the rank and update it.
    # But the rank is too large for floats.
    
    # Let's use the property: the middle sequence is the one that 
    # is lexicographically the largest sequence that is "smaller than or 
    # equal to" its own complement.
    # Wait, the simplest way is to use the symmetry:
    # The sequence at index (S+1)//2 is the one that, if you 
    # replace each x with N-x+1, you get the sequence at index S - (S+1)//2 + 1.
    # For S=6, (6+1)//2 = 3. Index 3 and Index 6-3+1 = 4 are mirrors.
    # The sequence at index 3 is the "largest" sequence that is 
    # "smaller" than its mirror.
    
    # Let's use a different approach: 
    # We can use a custom class to handle the large integers 
    # or use the property that we only need to compare 
    # (count of permutations with prefix) vs (S/2).
    # We can use `math.lgamma` to compare these values.
    # If lgamma(count) < log_S_half + epsilon, then we are still in the first half.
    
    # To avoid precision issues with lgamma, we can use a 
    # combination of lgamma for a rough check and a 
    # more precise method if they are very close.
    # But for N, K = 500, lgamma is precise enough to distinguish 
    # between different counts of permutations.
    
    # We need to track the current rank. Since we can't, 
    # we can use a target value `current_target_log` and 
    # update it by subtracting the log of the number of 
    # permutations we skip. 
    # Actually, we can't subtract in log space.
    # We must use the actual rank. 
    # But we can use `Decimal` or just use the fact that 
    # Python handles arbitrarily large integers.
    
    # Let's use the property: 
    # The number of permutations of a multiset is (sum n_i)! / product(n_i!).
    # We can compute this using `math.factorial`.
    
    def solve_with_big_ints():
        # Using a helper to calculate the number of permutations
        # We use a dictionary to memoize factorials
        fact = {0: 1}
        def f(n):
            if n not in fact:
                # This is a slow way to build the factorial table, 
                # but we only need it for values up to N*K
                pass
            return fact[n]

        # Precompute all factorials up to N*K
        import functools
        # Using a list comprehension to populate the fact dictionary
        # Since we can't use loops, we use map/reduce to create the factorial list
        facts = functools.reduce(lambda acc, _: acc + [acc[-1] * len(acc)], range(N * K), [1])
        
        def count_perms(counts):
            # counts is a list of remaining occurrences of each number
            # Total permutations = (sum(counts))! / product(c!)
            num = facts[sum(counts)]
            den = functools.reduce(lambda a, b: a * facts[b], counts, 1)
            return num // den

        # Total sequences S
        S = count_perms([K] * N)
        target = (S + 1) // 2
        
        # Current counts of each number 1...N
        rem = [K] * N
        result = []
        
        # We need to determine the element for each of the NK positions
        # To avoid loops, we use a recursive-like structure via reduce
        # But we need to maintain the state (rem, target)
        
        def get_element(state):
            current_rem, current_target = state
            # Find the smallest v such that sum_{i=1}^{v-1} count(i) < current_target
            # We can use a helper function to iterate through v=1...N
            def find_v(accumulator):
                # accumulator: (v, rank_sum, found_v)
                v, rank_sum, found_v = accumulator
                if found_v != -1:
                    return accumulator
                
                # Number of permutations if we pick number v+1
                # Remaining: rem[v] - 1, others same
                # Ways = (Total-1)! / ((rem[v]-1)! * product(rem[j]!))
                # Ways = [ (Total)! / product(rem[j]!) ] * rem[v] / Total
                
                # We can't use loops, so we use a comprehension to calculate 
                # the number of permutations for the current v
                # Wait, the number of permutations starting with v is:
                # (Total-1)! / ( (rem[v]-1)! * product_{j!=v}(rem[j]!) )
                # Which is: count_perms(rem) * rem[v] // Total
                
                # Since we are inside a function called by reduce, 
                # we can't easily loop. Let's use a different approach.
                return accumulator

        # Since the constraints are N, K <= 500, and we cannot use loops,
        # we can use a recursive function with a decorator for recursion limit.
        sys.setrecursionlimit(200000)
        
        def recurse(rem, target, pos):
            if pos == N * K:
                return []
            
            # Calculate total permutations for the current state
            # total_perms = count_perms(rem)
            # For v = 1 to N:
            #   perms_with_v = total_perms * rem[v-1] // (sum(rem))
            #   if target <= perms_with_v:
            #     return [v] + recurse(rem_with_v, target, pos + 1)
            #   else:
            #     target -= perms_with_v
            
            # To implement the "for v = 1 to N" without a loop:
            # We use a helper function and a range
            def search_v(v):
                # This is tricky without a loop. 
                # Let's use a generator and next()
                pass
        
        # Let's rethink. We can use a generator expression with next() 
        # to find the first v that satisfies the condition.
        
        def solve_recursive(current_rem, current_target, current_pos):
            if current_pos == N * K:
                return []
            
            # Total permutations of the current remaining set
            # Using the facts list precomputed outside
            total_rem = sum(current_rem)
            
            # We want the first v such that the sum of permutations 
            # for 1...v-1 is less than current_target.
            # Let f(v) be the number of permutations starting with v.
            #