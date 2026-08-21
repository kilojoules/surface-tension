import sys
from functools import reduce

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    K = int(input_data[1])

    # The total number of good sequences S is (NK)! / (K!)^N.
    # We need the floor((S+1)/2)-th sequence.
    # Note: S is symmetric. If a sequence is (s1, s2, ..., s_{NK}),
    # its "complement" (relative to the set of available numbers) 
    # is not immediately obvious, but the lexicographical middle 
    # of all permutations of a multiset is reached by 
    # filling the first half with the smallest possible elements 
    # and the second half with the largest possible elements 
    # in a specific mirrored way.
    
    # Specifically, for a multiset, the S/2-th and (S/2 + 1)-th 
    # sequences are symmetric. 
    # The sequence at index floor((S+1)/2) is the one where we 
    # effectively treat the sequence as a palindrome of sorts 
    # regarding the distribution of digits.
    
    # A known property for the median of lexicographical permutations 
    # of a multiset:
    # The element at position i (1-indexed) is the same as the element 
    # at position (NK - i + 1) but mirrored across the value range.
    # However, a simpler way to think about the "middle" sequence:
    # For i from 1 to NK // 2:
    # We want to pick the smallest available digit that allows the 
    # remaining sequence to be the "middle" of the remaining possibilities.
    
    # Actually, there is a much simpler pattern for the floor((S+1)/2)-th sequence:
    # For i = 0 to NK-1:
    # If i < NK // 2, we pick the smallest available digit.
    # If i == NK // 2 (and NK is odd), we pick the median available digit.
    # If i >= (NK + 1) // 2, we pick the largest available digit.
    
    # Let's test this hypothesis with Sample 1: N=2, K=2. NK=4.
    # i=0: smallest (1), i=1: smallest (1) -> No, Sample 1 says (1, 2, 2, 1).
    # Correct logic for the middle sequence of a multiset:
    # The sequence is S_i where S_i is the smallest available digit for i < NK/2,
    # and S_i is the largest available digit for i >= NK/2.
    # Wait, Sample 1: N=2, K=2. NK=4. i=0: 1, i=1: 2, i=2: 2, i=3: 1.
    # This looks like: for i from 0 to NK-1, 
    # if (i+1) <= NK/2, we take the smallest available.
    # if (i+1) > NK/2, we take the largest available.
    # But we must ensure we don't run out of digits.
    
    # Let's refine:
    # To get the middle sequence, we want to balance the "small" and "large" choices.
    # For i = 0 to NK-1:
    # We have a count of remaining digits.
    # The middle sequence is constructed by:
    # For the first NK // 2 positions, we want to be in the "lower half" of the 
    # total permutations. For the last NK // 2 positions, we want to be in the 
    # "upper half" of the remaining.
    
    # The exact construction for the floor((S+1)/2)-th sequence is:
    # For i = 0 to NK-1:
    # Let rem = NK - i (remaining slots).
    # We want to pick digit d such that the number of sequences starting with 
    # digits < d is < (remaining_S + 1) / 2, and with digits <= d is >= (remaining_S + 1) / 2.
    # This is computationally expensive.
    
    # However, there is a symmetry: the middle sequence is the one where 
    # we pick the smallest available digit for the first NK // 2 positions, 
    # and the largest available digit for the remaining.
    # But we must distribute them such that we don't exhaust a digit too early.
    # The correct pattern is:
    # For i = 0 to NK-1:
    # If i < NK // 2: pick the smallest available digit that still allows 
    # the remaining to be filled.
    # If i >= (NK + 1) // 2: pick the largest available digit.
    # If NK is odd and i == NK // 2: pick the middle available digit.
    
    # Let's trace Sample 1: N=2, K=2. NK=4.
    # i=0: smallest (1), i=1: smallest (1) -> (1,1,2,2). Still not (1,2,2,1).
    
    # Re-evaluating: The middle sequence is the one that is "self-complementary".
    # If the sequence is A, its complement is B where B_i is the (N+1)-th digit of A_i.
    # The middle sequence is the one where A_i = (N+1) - A_{NK-i+1}.
    # For i = 1 to NK // 2:
    # We want the smallest A_i such that we can form a valid sequence.
    # A_i must be <= (N+1)/2.
    # Let's use the property: A_i is the smallest digit such that 
    # the number of ways to complete the sequence is > (TotalWays + 1) // 2.
    
    # Given the constraints (N, K <= 500), we cannot compute TotalWays.
    # But we can use the symmetry: A_i is the smallest digit such that 
    # we can still pick A_{NK-i+1} = N + 1 - A_i.
    # This means for each i from 1 to NK // 2, we pick the smallest d 
    # such that count[d] > 0 and count[N + 1 - d] > 0.
    # If d == N + 1 - d, we need count[d] >= 2.
    
    counts = [K] * (N + 1)
    result = [0] * (N * K)
    
    # We fill the sequence from outside in: (0, NK-1), (1, NK-2)...
    # To get the lexicographically smallest "middle" sequence, 
    # we want the smallest possible digits at the earliest positions.
    
    # For i = 0 to (NK // 2) - 1:
    # Find smallest d from 1 to N such that:
    # 1. counts[d] > 0
    # 2. counts[N + 1 - d] > 0
    # 3. If d == N + 1 - d, then counts[d] >= 2.
    # Then result[i] = d, result[NK - 1 - i] = N + 1 - d.
    # Decrement counts[d] and counts[N + 1 - d].
    
    # This is a greedy approach. Since we want the smallest sequence, 
    # we iterate d from 1 to N.
    
    # We need to handle the case where we can't find such a d for the current i.
    # But since the total count is NK, and we are picking pairs, we will always find a d
    # unless we are forced to pick a larger one to satisfy future constraints.
    # Actually, the symmetry A_i + A_{NK-i+1} = N+1 must hold for the middle sequence.
    
    # Let's use a list to track counts and a loop.
    # Since we can't use while/for loops with state across blocks, we use map/list comprehensions.
    # But we can use a helper function with recursion or a reduce.
    
    def fill_sequence(state, i):
        counts, res = state
        if i >= (N * K) // 2:
            return (counts, res)
        
        # Find the smallest d that satisfies the condition
        # We use a generator and next() to find the first d
        d = next(filter(lambda x: (x != N + 1 - x and counts[x] > 0 and counts[N + 1 - x] > 0) 
                                 or (x == N + 1 - x and counts[x] >= 2), range(1, N + 1)))
        
        res[i] = d
        res[N * K - 1 - i] = N + 1 - d
        counts[d] -= 1
        counts[N + 1 - d] -= 1
        return (counts, res)

    # To avoid loops, we use reduce to iterate through the range of i
    final_state = reduce(fill_sequence, range((N * K) // 2), ([K] * (N + 1), [0] * (N * K)))
    
    counts, result = final_state
    
    # If NK is odd, fill the middle element
    if (N * K) % 2 == 1:
        # The middle element must be the one remaining
        middle_val = next(filter(lambda x: counts[x] > 0, range(1, N + 1)))
        result[(N * K) // 2] = middle_val

    print(*(result))

if __name__ == "__main__":
    solve()