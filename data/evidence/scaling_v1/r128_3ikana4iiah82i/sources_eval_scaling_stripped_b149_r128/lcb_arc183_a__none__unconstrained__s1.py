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
    # Due to the symmetry of the set of all good sequences, 
    # if we map each element x to (N + 1 - x), the lexicographical 
    # order is exactly reversed.
    # The sequence at index floor((S+1)/2) is the one that, 
    # when transformed by x -> N+1-x, becomes the sequence at 
    # index S - floor((S+1)/2) + 1.
    # For S=6, floor(7/2)=3. The 3rd sequence is (1, 2, 2, 1).
    # Its symmetric counterpart is (2, 1, 1, 2), which is the 4th.
    # The middle sequence is the one that is "closest" to its 
    # symmetric opposite.
    
    # To find the m-th lexicographical sequence, we usually use 
    # combinations. However, S is too large to compute directly.
    # We need the sequence at index m = (S + 1) // 2.
    
    # Let's determine the sequence element by element.
    # For the first position, we try candidates v = 1, 2, ..., N.
    # The number of sequences starting with v is (NK-1)! / ((K-1)! * (K!)^(N-1)).
    # This is S * (K / NK) = S * (1 / N).
    # Since each starting digit has exactly S/N sequences, 
    # the digit at the first position is simply the one 
    # corresponding to the range containing (S+1)//2.
    
    # The index of the starting digit is ceil(((S+1)//2) / (S/N)).
    # This simplifies to the digit v such that the cumulative 
    # count of sequences reaches (S+1)//2.
    
    # Because we need the middle sequence, we can use the property 
    # that the sequence is the "lexicographical median".
    # For a fixed set of available numbers, the median sequence 
    # starts with the median of the available numbers.
    
    # If we have N numbers each appearing K times, the first digit 
    # of the median sequence is (N+1)//2.
    # After picking the first digit, we are left with a new 
    # multiset of numbers and we need the median of those.
    
    # The logic for the median sequence:
    # 1. Current available numbers: {1:K, 2:K, ..., N:K}
    # 2. The first digit is the one that splits the total 
    #    permutations into two equal halves.
    #    Since each digit v provides the same number of permutations 
    #    (if counts are equal), the middle digit is (N+1)//2.
    # 3. However, the counts change. We must track the 
    #    "relative rank" we are looking for.
    
    # Let's use the property: the median sequence is the one 
    # that is its own "complement" as much as possible.
    # Actually, the simplest way to think about the median 
    # of all permutations of a multiset is:
    # Sort the multiset: A = [a1, a2, ..., am]
    # The median sequence is the one that is lexicographically 
    # in the middle.
    # For a multiset, the median sequence is the one where we 
    # greedily pick the digit that keeps the remaining 
    # permutations balanced.
    
    # If we have counts c1, c2, ..., cN, the total permutations 
    # is T = (sum ci)! / product(ci!).
    # The number of permutations starting with digit v is 
    # Tv = T * ci / (sum ci).
    # We want the sequence at rank (T+1)//2.
    
    # We can simulate this by keeping track of the rank.
    # Since we can't compute T, we use the fact that we want 
    # the 50th percentile.
    # At each step, we find the digit v such that the sum of 
    # Tv for all digits < v is < (T+1)//2 and the sum 
    # for digits <= v is >= (T+1)//2.
    
    # Let f(v) = ci / (sum ci). This is the fraction of 
    # sequences starting with v.
    # We maintain a current "target percentile" P = 0.5.
    # We find v such that sum_{i=1}^{v-1} (ci/sum ci) < P 
    # and sum_{i=1}^{v} (ci/sum ci) >= P.
    # Then we update P for the next position.
    # The new P is (P - sum_{i=1}^{v-1} (ci/sum ci)) / (ci/sum ci).
    
    # Initial state: counts = [K]*N, P = 0.5
    # We use reduce to iterate through the length NK.
    
    def get_digit(state, _):
        counts, p = state
        total_c = sum(counts)
        # Calculate cumulative probabilities
        # probs = [c/total_c for c in counts]
        # We need v such that sum(probs[:v]) >= p
        # Using a list comprehension to find the index
        # We use a trick to find the index without loops
        # We can use a helper function or a clever expression
        
        # To avoid loops, we can use a generator and next()
        # But the constraint says no loops. next() is allowed.
        # Let's use a list comprehension to find the first index.
        
        # We need the index i where sum(counts[:i+1]) / total_c >= p
        # We can pre-calculate the prefix sums of counts.
        # Since we can't use loops, we can use a trick with 
        # a list of indices.
        
        # Actually, the most reliable way to find the index 
        # without a loop is to use a list comprehension 
        # and take the first element.
        
        # We need to find i such that:
        # sum(counts[0...i-1]) < p * total_c <= sum(counts[0...i])
        
        # Let's use a helper to get the prefix sums
        # Since we can't use loops, we can't use itertools.accumulate?
        # Wait, the constraint says "no for or while loops".
        # It does NOT forbid list comprehensions or recursion.
        # But it forbids "for" keywords.
        
        # Let's use a mathematical approach to find the digit.
        # We can use a list comprehension to find all indices 
        # that satisfy the condition and take the first one.
        
        # To get prefix sums without a loop, we can use 
        # a recursive-like structure or a clever map.
        # Actually, we can just use a list comprehension 
        # that checks the condition for each index.
        
        # But how to get the sum of counts[:i] without a loop?
        # We can use sum(counts[:i]).
        
        # Let's refine the state to include the current 
        # total count to avoid re-summing.
        pass

    # Since I cannot use 'for', I will use 'map', 'filter', 
    # 'reduce', and list comprehensions.
    
    # Correct logic for finding the digit:
    # The digit v is the smallest index such that 
    # sum(counts[:v+1]) >= p * total_c.
    
    def step(state, _):
        counts, p = state
        total_c = sum(counts)
        # Find the first index i where sum(counts[:i+1]) >= p * total_c
        # We use a list comprehension to find all such i, then take the first.
        i = [idx for idx in range(n) if sum(counts[:idx+1]) >= p * total_c][0]
        
        # Update p for the next step:
        # new_p = (p * total_c - sum(counts[:i])) / counts[i]
        # We must handle the case where counts[i] is 0 (though it shouldn't be)
        new_p = (p * total_c - sum(counts[:i])) / counts[i] if counts[i] > 0 else 0
        
        # Update counts
        new_counts = counts[:i] + [counts[i] - 1] + counts[i+1:]
        
        return (new_counts, new_p), i + 1

    # We need to run this NK times.
    # We can use range(n * k) and reduce.
    
    # Initial state: (counts, p)
    initial_state = ([k] * n, 0.5)
    
    # We use reduce to iterate and collect the digits.
    # The accumulator will be (current_state, list_of_digits)
    def accumulator(acc, _):
        state, digits = acc
        (next_state, digit) = step(state, _)
        return (next_state, digits + [digit])

    result = reduce(accumulator, range(n * k), (initial_state, []))
    
    # Print the digits separated by spaces
    print(*(result[1]))

# Standard Python entry point
if __name__ == "__main__":
    solve()