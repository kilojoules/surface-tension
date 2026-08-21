```python
import sys
from functools import reduce

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    K = int(input_data[1])

    # The total number of good sequences S is (NK)! / (K!^N).
    # We need the floor((S+1)/2)-th sequence.
    # Note: S is symmetric. If a sequence is (a1, a2, ..., aM), 
    # its "complement" relative to the sorted list of all elements is 
    # (N+1-a1, N+1-a2, ..., N+1-aM).
    # However, the problem asks for the middle of the lexicographical order.
    # A key property of these sequences is that if you replace each x with (N+1-x),
    # you get another good sequence.
    # The lexicographically smallest sequence is (1*K, 2*K, ..., N*K).
    # The lexicographically largest sequence is (N*K, (N-1)*K, ..., 1*K).
    # These two are "dual" to each other.
    # The floor((S+1)/2)-th sequence is the one immediately preceding the 
    # exact middle of the list.
    # Due to the symmetry of the construction, the sequence we are looking for
    # is the one where we try to pick the smallest possible number at each step,
    # but we must ensure that the remaining count of sequences is at least half of S.
    # Actually, a simpler way to think about it:
    # We want the sequence X such that the number of sequences < X is floor((S-1)/2).
    # Since we cannot compute S (it's too large), we use the property:
    # The "middle" sequence is reached when we have exhausted half of the total permutations.
    # Because of the symmetry (replacing i with N+1-i), the sequence at index S/2 
    # starts with the smallest possible digits until it's forced to "flip" to the 
    # larger half.
    # Specifically, the floor((S+1)/2)-th sequence is the one that is 
    # lexicographically smaller than or equal to its "dual" (where each x is replaced by N+1-x).
    # The sequence that is exactly in the middle (or the smaller of the two middle ones)
    # is the one that starts with the smallest possible digits such that the 
    # remaining suffix is the "largest" possible sequence that doesn't push us 
    # into the second half of the total set.
    
    # Correct logic for "middle" of symmetric combinatorial sets:
    # The sequence is formed by: for each position, try digits d = 1, 2, ..., N.
    # If we pick d, the number of ways to complete the sequence is W.
    # If the current rank we are looking for is > W, we subtract W from rank and try d+1.
    # To find the floor((S+1)/2)-th sequence without calculating S:
    # We can maintain the current counts of each number.
    # At each step, we want to know if the number of sequences starting with the 
    # current prefix is less than half of the total sequences possible with the 
    # remaining counts.
    # Let f(c1, c2, ..., cN) be the number of ways to arrange the remaining characters.
    # f = (sum(ci))! / product(ci!)
    # We want to pick d such that sum_{i < d} f(c1, ..., ci-1, ..., cN) < S/2 <= sum_{i <= d} f(...)
    # This is equivalent to: we pick the smallest d such that the number of sequences 
    # starting with digits < d is strictly less than S/2.
    # S/2 = (1/2) * (sum(ci)! / product(ci!))
    # The condition "sum_{i < d} f(c1, ..., ci-1, ..., cN) < S/2" can be checked by 
    # comparing the sum to S/2, or 2 * sum < S.
    # S = (sum(cj))! / product(cj!)
    # f(c1, ..., ci-1, ..., cN) = S * (ci / sum(cj))
    # So we want sum_{i < d} (ci / sum(cj)) < 1/2.
    # This simplifies to: sum_{i < d} ci < (1/2) * sum_{j=1}^N cj.
    # In other words, sum_{i < d} ci < sum_{i >= d} ci.
    
    # Let's refine:
    # At each position, we have counts c1, c2, ..., cN.
    # We want to find the smallest d such that the number of sequences starting with 
    # 1, ..., d-1 is strictly less than S/2.
    # The number of sequences starting with 1, ..., d-1 is:
    # Total_Remaining_Sequences * ( (sum_{i=1}^{d-1} ci) / (sum_{j=1}^N cj) )
    # We want: (sum_{i=1}^{d-1} ci) / (sum cj) < 1/2
    # => 2 * (sum_{i=1}^{d-1} ci) < sum_{j=1}^N cj
    
    # Wait, this is only true for the first digit. For subsequent digits, 
    # the "halfway" point shifts.
    # Let's use the property: the floor((S+1)/2)-th sequence is the one that is 
    # "just smaller" than its dual.
    # The dual of (S1, ..., SM) is (N+1-S1, ..., N+1-SM).
    # A sequence is <= its dual if at the first index i where they differ, Si < N+1-Si.
    # This means Si < (N+1)/2.
    # To get the largest sequence that is <= its dual:
    # For the first index i where we can make a choice, we want to pick the largest 
    # possible digit that still allows the sequence to be <= its dual.
    # If we pick Si < (N+1)/2, then for all j > i, we want Sj to be as large as possible.
    # If we pick Si > (N+1)/2, then the sequence is > its dual.
    # If we pick Si = (N+1)/2 (only if N is odd), we move to the next index.
    
    # Correct logic for floor((S+1)/2)-th:
    # The set of all good sequences can be paired (X, dual(X)).
    # If X < dual(X), then X comes before dual(X).
    # If X = dual(X), then X is the exact middle.
    # The floor((S+1)/2)-th sequence is the largest X such that X <= dual(X).
    # To construct the largest X such that X <= dual(X):
    # For each position p = 1 to NK:
    # We want to pick the largest d in {1, ..., N} such that:
    # 1. count[d] > 0
    # 2. There exists some way to complete the sequence such that X <= dual(X).
    # If we pick d < (N+1)/2, then X < dual(X) regardless of the rest. To maximize X,
    # we then pick the largest possible remaining digits for the rest of the sequence.
    # If we pick d > (N+1)/2, then X > dual(X) regardless of the rest. This is forbidden.
    # If we pick d = (N+1)/2, we are still in the "undecided" state.
    
    # Let's trace:
    # For p = 1 to NK:
    #   We want to see if we can pick a digit d > (N+1)/2. 
    #   But if we do, X becomes > dual(X). 
    #   So we can only pick d > (N+1)/2 if we have already picked some d' < (N+1)/2 
    #   at an earlier position.
    #   If we haven't picked any d' != (N+1)/2 yet:
    #     We can pick d < (N+1)/2. If we do, the rest of the sequence should be 
    #     lexicographically largest (descending order).
    #     We can pick d = (N+1)/2 (if N is odd). Then we continue to the next position.
    #     We cannot pick d > (N+1)/2.
    
    # This means:
    # While we haven't picked a digit other than (N+1)/2:
    #   Try d = N, N-1, ..., 1.
    #   If d > (N+1)/2: cannot pick (would make X > dual(X)).
    #   If d == (N+1)/2: we can pick it and stay "undecided".
    #   If d < (N+1)/2: we can pick it, and then all subsequent digits are largest possible.
    
    # But we want the largest X such that X <= dual(X).
    # So at each step, we first try to pick d = (N+1)/2.
    # If we can't, or if we want to see if we can pick something larger, 
    # we can't pick d > (N+1)/2 unless we already picked some d' < (N+1)/2.
    # Wait, if we pick d < (N+1)/2, we can then pick the largest possible for the rest.
    # If we pick d = (N+1)/2, we are still restricted.
    # To maximize X:
    # At each step, we prefer d = (N+1)/2 over d < (N+1)/2.
    # Because if we pick d < (N+1)/2, we can fill the rest with largest, 
    # but if we pick d = (N+1)/2, we might be able to pick an even larger digit later.
    
    # Let's refine:
    # For p = 1 to NK:
    #   If already_smaller:
    #     Pick largest d with count[d] > 0.
    #   Else:
    #     Try d = (N+1)//2 (if N is odd and count[d] > 0).
    #     If we can't or we want to check if we can get a larger sequence:
    #     Actually, the largest X <= dual(X) is:
    #     For each position, try d = N, N-1, ..., 1.
    #     If d > (N+1)/2:
    #       This is only allowed if already_smaller is True.
    #     If d == (N+1)/2:
    #       This is always allowed. It doesn't change already_smaller.
    #     If d < (N+1)/2:
    #       This is always allowed. It sets already_smaller = True.
    
    # To maximize X, at each step we try d = N, N-1, ..., 1.
    # If d > (N+1)/2:
    #   If already_smaller: pick d, continue.
    #   Else: cannot pick d.
    # If d == (N+1)/2:
    #   Pick d, continue.
    # If d < (N+1)/2:
    #   We can pick d, and then all remaining are largest.
    #   But we should first check if we can pick d = (N+1)/2 and potentially 
    #   get a larger sequence.
    #   Actually, if we pick d < (N+1)/2, we immediately set already_smaller = True.
    #   The largest possible sequence starting with d < (N+1)/2 is (d, then all remaining descending).
    #   The largest possible sequence starting with d = (N+1)/2 is (d, then recurse).
    #   Since (N+1)/2 > d, we always prefer d = (N+1)/2 if available.
    
    counts = [K] * N
    result = []
    already_smaller = False
    
    # We need to simulate the process for NK steps.
    # Since we can't use loops, we use a recursive-like structure or a list comprehension.
    # But we can use a mutable state and a loop replaced by a reduce or a custom function.
    
    state = {'already_smaller': False, 'counts': counts}
    
    def get_digit(pos):
        s = state
        c = s['counts']
        is_small = s['already_smaller']
        
        # Try digits from N down to 1
        # We use a generator to find the first