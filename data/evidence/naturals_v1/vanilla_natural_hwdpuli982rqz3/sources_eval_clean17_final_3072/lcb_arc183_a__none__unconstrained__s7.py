import sys
from functools import reduce

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    K = int(input_data[1])

    # The total number of good sequences S is (NK)! / (K!^N)
    # We need the floor((S+1)/2)-th sequence.
    # Note: S is symmetric. If we have a sequence (a1, a2, ..., aM),
    # its "complement" (N+1-a1, N+1-a2, ..., N+1-aM) is also a good sequence.
    # Lexicographically, if A < B, then complement(A) > complement(B).
    # The middle two sequences (if S is even) are A and complement(A) where A is the 
    # largest sequence starting with a number <= N/2.
    # Actually, a simpler property: the floor((S+1)/2)-th sequence is the 
    # lexicographically last sequence that is "smaller than or equal to" its 
    # complement (where complement replaces x with N+1-x).
    # However, the most direct way to find the middle is to realize that 
    # the set of all sequences is symmetric around the "center".
    # The floor((S+1)/2)-th sequence is the one immediately preceding the 
    # sequence that would be the "exact middle".
    # Due to symmetry, the floor((S+1)/2)-th sequence is the largest sequence 
    # that starts with a digit 'd' such that d < (N+1)/2, 
    # or if N is odd, it might start with (N+1)//2.
    
    # Correct logic for symmetry:
    # Let the sorted good sequences be X_1, X_2, ..., X_S.
    # X_i is the complement of X_{S-i+1}.
    # We want X_{floor((S+1)/2)}.
    # If we can determine if a sequence is "smaller than its complement", 
    # we can binary search or greedily build it.
    # A sequence A is smaller than its complement A' if at the first index i where A_i != A'_i, A_i < A'_i.
    # A'_i = N + 1 - A_i.
    # So A_i < N + 1 - A_i  => 2 * A_i < N + 1.
    
    # To find the floor((S+1)/2)-th sequence:
    # This is the largest sequence A such that A <= complement(A).
    # For each position i from 1 to NK:
    # We try digits d from 1 to N (that still have remaining counts).
    # We need to check if there exists any sequence starting with (prefix + d) 
    # that is <= its complement.
    # If we pick d < (N+1)/2, then for all subsequent choices, the sequence 
    # will be smaller than its complement regardless of what follows.
    # If we pick d > (N+1)/2, then the sequence will be larger than its complement.
    # If we pick d = (N+1)/2 (only possible if N is odd), we move to the next index.
    
    # Greedy strategy to find the largest A such that A <= complement(A):
    # For each position i:
    # 1. Try d from N down to 1.
    # 2. If d < (N+1)/2: 
    #    This d makes the sequence A < complement(A). 
    #    Since we want the largest such A, we take the largest d < (N+1)/2, 
    #    and then fill the rest of the sequence with the largest possible digits.
    # 3. If d > (N+1)/2:
    #    This d makes A > complement(A). We cannot pick this unless we already 
    #    established A < complement(A) at a previous index.
    # 4. If d == (N+1)/2:
    #    This doesn't decide yet. We move to the next index.
    
    # Let's refine:
    # At index i, we have a state: "Already decided A < A'" (True/False).
    # If decided == True: pick largest available d.
    # If decided == False:
    #    Can we pick d > (N+1)/2? 
    #    If we do, then A becomes > A'. This is only allowed if we can 
    #    later make A < A', but the first difference determines it.
    #    So if decided == False, we cannot pick d > (N+1)/2.
    #    We can pick d < (N+1)/2, then decided becomes True.
    #    We can pick d == (N+1)/2, then decided stays False.
    
    # To get the largest A <= A':
    # For i = 0 to NK-1:
    #   For d = N down to 1:
    #     If count[d] > 0:
    #       If decided == True:
    #         (This d is fine, and since we go from N down to 1, it's the largest)
    #         ans[i] = d, count[d]--, break
    #       Else (decided == False):
    #         If d < (N+1)/2:
    #           (This d makes A < A'. This is the largest d that triggers 'True')
    #           # But wait, we should first check if we can pick d == (N+1)/2 
    #           # and still find a valid sequence.
    #           # If we pick d < (N+1)/2, we can fill the rest with the absolute maximums.
    #           # If we pick d == (N+1)/2, we continue restricting.
    #           # To maximize the sequence, we prefer d == (N+1)/2 over d < (N+1)/2.
    
    # Correct Greedy:
    # For i = 0 to NK-1:
    #   1. If decided == True:
    #      Pick largest available d.
    #   2. If decided == False:
    #      If N is odd and count[(N+1)//2] > 0:
    #        We can potentially pick d = (N+1)//2 and stay in decided = False.
    #        But we must ensure that it's possible to complete the sequence.
    #        (It always is, as long as we have numbers left).
    #        However, we want the largest sequence. 
    #        If we pick d = (N+1)//2, we are more restricted than if we pick d < (N+1)//2? 
    #        No, picking a larger number at the current index is always better.
    #        So if N is odd and count[(N+1)//2] > 0, we take it and stay decided = False.
    #        Then we check if we can take any d < (N+1)//2.
    #        Wait, if we take d < (N+1)//2, we immediately set decided = True and 
    #        fill the rest with the maximums.
    #        We compare:
    #        Option A: Pick d = (N+1)//2, then continue greedily.
    #        Option B: Pick d = (N+1)//2 - 1, then fill rest with max.
    #        Since (N+1)//2 > (N+1)//2 - 1, Option A is better.
    
    # Revised Greedy:
    # For i = 0 to NK-1:
    #   If decided:
    #     Pick largest available d.
    #   Else:
    #     If N is odd and count[(N+1)//2] > 0:
    #       ans[i] = (N+1)//2, count[(N+1)//2]--
    #     Else:
    #       Pick largest available d < (N+1)/2.
    #       decided = True
    #       Fill remaining with largest available.
    
    # Let's trace Sample 1: N=2, K=2. S=6, floor(7/2)=3.
    # i=0: decided=F. N=2 (even). Largest d < 1.5 is 1.
    # ans[0]=1, decided=T.
    # i=1: decided=T. Largest available is 2. ans[1]=2.
    # i=2: decided=T. Largest available is 2. ans[2]=2.
    # i=3: decided=T. Largest available is 1. ans[3]=1.
    # Result: 1 2 2 1. Correct.
    
    # Sample 4: N=3, K=3. S=1680/6=280. floor(281/2)=140.
    # i=0: decided=F. N=3 (odd). count[2]=3 > 0. ans[0]=2, count[2]=2.
    # i=1: decided=F. N=3 (odd). count[2]=2 > 0. ans[1]=2, count[2]=1.
    # i=2: decided=F. N=3 (odd). count[2]=1 > 0. ans[2]=2, count[2]=0.
    # i=3: decided=F. N=3 (odd). count[2]=0. Largest d < 2 is 1.
    # ans[3]=1, count[1]=2, decided=T.
    # i=4: decided=T. Largest available is 3. ans[4]=3, count[3]=2.
    # i=5: decided=T. Largest available is 3. ans[5]=3, count[3]=1.
    # i=6: decided=T. Largest available is 3. ans[6]=3, count[3]=0.
    # i=7: decided=T. Largest available is 1. ans[7]=1, count[1]=1.
    # i=8: decided=T. Largest available is 1. ans[8]=1, count[1]=0.
    # Result: 2 2 2 1 3 3 3 1 1. Correct.

    counts = [K] * (N + 1)
    ans = [0] * (N * K)
    decided = False
    
    # We use a while loop with an index to simulate the process
    idx = 0
    while idx < N * K:
        if decided:
            # Pick largest available
            for d in range(N, 0, -1):
                if counts[d] > 0:
                    ans[idx] = d
                    counts[d] -= 1
                    break
            idx += 1
        else:
            # Try to stay undecided by picking (N+1)//2
            mid = (N + 1) // 2
            if N % 2 != 0 and counts[mid] > 0:
                ans[idx] = mid
                counts[mid] -= 1
                idx += 1
            else:
                # Must decide now. To maximize, pick largest d < (N+1)/2
                # Note: (N+1)//2 is the threshold.
                # If N=2, mid=1. Largest d < 1.5 is 1.
                # If N=3, mid=2. Largest d < 2 is 1.
                # The largest d such that d < (N+1)/2 is (N // 2).
                d = N // 2
                ans[idx] = d
                counts[d] -= 1
                decided = True
                idx += 1
                
    print(*(ans))

if __name__ == "__main__":
    solve()