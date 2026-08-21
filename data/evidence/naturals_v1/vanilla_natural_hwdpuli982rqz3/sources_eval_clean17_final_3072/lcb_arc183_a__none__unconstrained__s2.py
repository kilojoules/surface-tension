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

    # The total number of good sequences S is (NK)! / (K!)^N.
    # We need the floor((S+1)/2)-th sequence.
    # Note: S is symmetric. If a sequence is (s1, s2, ..., s_{NK}),
    # its "complement" (relative to the sorted list of available numbers)
    # is also a good sequence.
    # Specifically, if we replace each element x with (N + 1 - x), 
    # we get another good sequence.
    # Lexicographically, the middle two sequences (if S is even) are:
    # The last sequence starting with 1 (or the smallest possible prefix)
    # and the first sequence starting with 2 (or the largest possible prefix).
    # Actually, a simpler property:
    # The S/2-th sequence is the largest sequence that is "smaller" than its 
    # complement (where complement is replacing x with N+1-x).
    # However, the most direct way to find the floor((S+1)/2)-th sequence
    # is to realize that the set of all good sequences is symmetric.
    # The "middle" of the lexicographical order is reached when we 
    # transition from sequences starting with 1... to sequences starting with 2...
    # and so on.
    
    # Let's use the property: The floor((S+1)/2)-th sequence is the 
    # lexicographically last sequence X such that X <= complement(X).
    # Where complement(X) is obtained by replacing each x_i with (N + 1 - x_i).
    # If X = complement(X), then X is the exact middle.
    
    # To construct the last X such that X <= complement(X):
    # At each position i, we want to pick the largest possible digit d
    # such that the resulting prefix can still be completed to a sequence X 
    # where X <= complement(X).
    
    # Let the current prefix be P. Let the complement prefix be P'.
    # If P < P', then for the remaining positions, we can pick the largest 
    # possible digits to make X as large as possible, and it will still be < P'.
    # If P > P', then this prefix is already too large.
    # If P = P', we must continue picking d such that d <= (N + 1 - d).
    
    # State: (counts of remaining numbers, is_less)
    # counts: tuple of length N
    # is_less: boolean (True if P < P', False if P = P')
    
    # Since we can't use recursion/DP easily with loops, we use a greedy approach.
    # We maintain the counts of remaining numbers.
    counts = [K] * N
    is_less = False
    result = []
    
    # We need to fill NK positions
    for i in range(N * K):
        # Try digits d from N down to 1
        for d in range(N, 0, -1):
            if counts[d-1] > 0:
                # Check if picking d maintains X <= complement(X)
                # Complement of d is (N + 1 - d)
                comp_d = N + 1 - d
                
                # If we are already 'less', any valid d is fine.
                # If we are 'equal', we need d <= comp_d.
                # Exception: if we pick d > comp_d, we become 'greater', which is forbidden.
                # If we pick d < comp_d, we become 'less'.
                
                if not is_less and d > comp_d:
                    continue
                
                # If we pick d, will we be able to complete the sequence?
                # Yes, as long as we have numbers left.
                # But we must ensure that if we are still 'equal', 
                # we don't force ourselves into a 'greater' state later.
                # Actually, if is_less is True, we just take the largest possible.
                # If is_less is False, we take the largest d <= comp_d.
                # If we take d < comp_d, is_less becomes True.
                
                # To maximize X, we want to transition to is_less = True as late as possible.
                # But we can't just pick d = comp_d every time because we might run out of comp_d.
                
                # Correct logic for "Last X such that X <= complement(X)":
                # At each step, try d = N, N-1, ..., 1.
                # If is_less is True: take the largest d available.
                # If is_less is False: 
                #    If d < comp_d: this is possible, and is_less becomes True.
                #    If d == comp_d: this is possible, and is_less stays False.
                #    If d > comp_d: not possible.
                
                # However, we must ensure that if we pick d == comp_d, 
                # there is still a way to complete the sequence.
                # Since we only need to check if a valid sequence exists (which it does 
                # as long as counts > 0), the only constraint is d <= comp_d when is_less is False.
                
                # To get the ABSOLUTE last sequence, we should first try to keep is_less = False
                # by picking d = comp_d, and if that's not possible, pick the largest d < comp_d.
                
                # Let's refine:
                # If is_less: pick max d.
                # If not is_less:
                #    Can we pick d = comp_d? 
                #    Only if counts[comp_d-1] > 0.
                #    If we do, is_less remains False.
                #    If we pick d < comp_d, is_less becomes True.
                #    To maximize X, we prefer d = comp_d over d < comp_d.
                
                # Wait, if we pick d < comp_d, we can then pick the largest possible for all remaining.
                # If we pick d = comp_d, we are still constrained.
                # Example N=2, K=2: S=6, floor(7/2)=3. Sequences: 1122, 1212, 1221, 2112, 2121, 2211.
                # 3rd is 1221.
                # i=0: d=2? comp_d=1. 2 > 1, no. d=1? comp_d=2. 1 < 2, is_less=True.
                # i=1: is_less=True, max d=2.
                # i=2: is_less=True, max d=2.
                # i=3: is_less=True, max d=1.
                # Result: 1 2 2 1. Correct.
                
                # Let's re-verify: if we have the option to pick d = comp_d or d < comp_d.
                # If we pick d = comp_d, we stay at is_less = False.
                # If we pick d < comp_d, we move to is_less = True.
                # Once is_less = True, we can pick the maximum possible for all remaining.
                # This will always be lexicographically larger than staying at is_less = False.
                # No, that's wrong. If we pick d < comp_d, the digit at this position is smaller.
                # To maximize the sequence, we want the largest digit at the earliest position.
                # So if is_less is False:
                # 1. Try d = comp_d. If possible, we stay at is_less = False.
                # 2. If we can't or choose not to, try d < comp_d. Then is_less becomes True.
                # But we want the largest sequence. So we should try d = comp_d first.
                # If we take d = comp_d, we are constrained. If we take d < comp_d, we are free.
                # But d < comp_d is smaller than d = comp_d.
                # So we should try d = comp_d first. If we can complete the sequence with 
                # the remaining counts such that X <= comp(X), then that's better.
                
                # Actually, if we pick d = comp_d, we just need to check if the remaining 
                # counts can form a sequence Y such that Y <= comp(Y).
                # This is always possible (e.g., the smallest possible sequence).
                
                # Correct Greedy:
                # For i = 0 to NK-1:
                #   For d = N down to 1:
                #     if counts[d-1] > 0:
                #       if is_less:
                #         # We can take this d, and is_less remains True
                #         # Since we iterate d from N down to 1, the first valid d is the max.
                #         # But we must ensure the remaining can be filled. (Always true if counts > 0)
                #         # Wait, if is_less is True, we just take the largest d and move on.
                #         # But we must be careful: if we take a very large d, we might 
                #         # exhaust a number that is needed to satisfy X <= comp(X) later?
                #         # No, if is_less is already True, X is already < comp(X).
                #         # The remaining digits can be anything. To maximize X, take max d.
                #         pass 
                #       elif d < comp_d:
                #         # This makes is_less = True. This is possible.
                #         # But we should first check if d = comp_d is possible.
                #         pass
                #       elif d == comp_d:
                #         # This keeps is_less = False.
                #         pass
                #       else: # d > comp_d
                #         # Not allowed since is_less is False.
                #         continue
                
                # Let's refine the "is_less" logic:
                # If is_less is True:
                #    Pick largest d available.
                # If is_less is False:
                #    1. Try d = comp_d. If available, we can take it and stay is_less = False.
                #       But wait, if we take d = comp_d, we are more constrained than if we took d < comp_d.
                #       However, d = comp_d is larger than any d < comp_d.
                #       So we should take d = comp_d if it's possible to complete the sequence.
                #       Can we always complete it? Yes, by picking the smallest remaining.
                #    2. If we can't take d = comp_d, or we want to see if we can get a larger sequence,
                #       we try d < comp_d. This makes is_less = True, and we can then pick the max for all rest.
                
                # Let's trace N=2, K=2 again.
                # i=0: d=2, comp_d=1. 2 > 1, fail. d=1, comp_d=2. 1 < 2, is_less=True.
                # i=1: is_less=True, d=2, ok.
                # i=2: is_less=True, d=2, ok.
                # i=3: is_less=True, d=1, ok.
                # Result: 1 2 2 1.
                
                # Let's trace N=3, K=1. S=6, floor(7/2)=3. 123, 132, 213, 231, 312, 321. 3rd is 213.
                # i=0: d=3, comp_d=1. 3>1 fail. d=2, comp_d=2. 2==2, is_less=False.
                # i=1: d=3, comp_d=1. 3>1 fail. d=2, 0 left. d=1, comp_d=3. 1<3, is_less=True.
                # i=2: is_less=True, d=3, ok.
                # Result: 2 1 3. Correct.
                
                # The only catch: if we pick d = comp_d, we must ensure that the remaining 
                # counts can actually form a sequence Y such that Y <= comp(Y).
                # This is possible if and only if for all x, count(