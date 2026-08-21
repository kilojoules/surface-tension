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
    # We want the floor((S+1)/2)-th sequence.
    # Note: S is symmetric. If we have a sequence (a_1, ..., a_{NK}),
    # its "complement" (N+1-a_1, ..., N+1-a_{NK}) is also a good sequence.
    # Lexicographically, if A < B, then complement(A) > complement(B).
    # The middle two sequences (if S is even) or the middle one (if S is odd)
    # are related by this symmetry.
    # Specifically, the floor((S+1)/2)-th sequence is the one immediately 
    # preceding the "halfway" point.
    # Due to the symmetry of the set of all permutations of the multiset,
    # the sequence at index floor((S+1)/2) is the one that is 
    # lexicographically largest among those that start with a digit <= N/2,
    # or more simply: it is the sequence that is "just smaller" than its 
    # complement.
    
    # Actually, a simpler observation:
    # The set of all good sequences is symmetric around the "middle".
    # The sequence at rank floor((S+1)/2) is the one where we try to pick 
    # the smallest possible numbers for the first half of the sequence 
    # and the largest for the second half, but balanced.
    # More formally: the sequence X is the floor((S+1)/2)-th if 
    # X is the largest sequence such that X <= complement(X).
    # This means at the first index i where X_i != complement(X)_i, we must have X_i < complement(X)_i.
    # Since complement(X)_i = N + 1 - X_i, this means X_i < (N + 1) / 2.
    
    # To find the largest X such that X <= complement(X):
    # We want to maximize X. To keep X <= complement(X), we must ensure that 
    # at the first position i where they differ, X_i < N + 1 - X_i.
    # To make X largest, we want to push this "differing" position as far back as possible.
    # For all j < i, we must have X_j = N + 1 - X_j, which implies 2*X_j = N + 1.
    # This is only possible if N is odd and X_j = (N+1)/2.
    
    # Let M = (N + 1) // 2.
    # If N is even, the first elements must differ eventually. To maximize X,
    # we want X_1 to be as large as possible but still allow X <= complement(X).
    # The condition X <= complement(X) is satisfied if the first index i where 
    # X_i != N+1-X_i satisfies X_i < N+1-X_i.
    
    # To maximize X:
    # 1. Fill positions with (N+1)/2 as long as N is odd and we have K left.
    # 2. At the first position where we cannot put (N+1)/2 (or if N is even),
    #    we want to put the largest possible value 'v' such that v < (N+1)/2.
    #    That is v = (N // 2).
    # 3. Once we have placed a value v < (N+1)/2, the condition X < complement(X) 
    #    is permanently satisfied. To maximize the rest of the sequence, 
    #    we fill the remaining positions with the largest available numbers.
    
    # Let's refine:
    # The sequence X that is floor((S+1)/2)-th is the one where:
    # - For i = 1 to NK:
    #   - If we can pick x such that x = N+1-x (i.e., x = (N+1)/2), we do so? 
    #   - No, to maximize X, we want the first difference to be as late as possible.
    #   - While we can, we pick X_i = (N+1)/2 (if N is odd).
    #   - Then we pick X_i = N // 2.
    #   - Then we fill the rest in descending order.
    
    # Wait, if N=2, K=2. S=6. floor(7/2)=3. Sequences: (1,1,2,2), (1,2,1,2), (1,2,2,1)...
    # N=2, K=2 -> (1, 2, 2, 1).
    # My logic: N is even, so we can't pick (N+1)/2. First element X_1 = N//2 = 1.
    # Remaining: {1:1, 2:2}. Fill descending: 2, 2, 1. Result: (1, 2, 2, 1). Correct.
    
    # Sample 3: N=6, K=1. S=720. floor(721/2)=360.
    # N is even. X_1 = N//2 = 3. Remaining {1,2,4,5,6}. Descending: 6, 5, 4, 2, 1.
    # Result: (3, 6, 5, 4, 2, 1). Correct.
    
    # Sample 4: N=3, K=3. S=1680/6=280. floor(281/2)=140.
    # N is odd. (N+1)/2 = 2. 
    # We can pick X_i = 2 for i=1, 2, 3.
    # Then we must pick X_4 < (N+1)/2. The largest such is 1.
    # Remaining: {1:2, 3:3}. Descending: 3, 3, 3, 1, 1.
    # Result: (2, 2, 2, 1, 3, 3, 3, 1, 1). Correct.
    
    # General Algorithm:
    # 1. If N is odd, let mid = (N + 1) // 2. Output 'mid' K times.
    # 2. Let low = N // 2. Output 'low' once.
    # 3. Output all remaining numbers in descending order.
    #    The remaining numbers are:
    #    - 'low' (K-1 times)
    #    - 'low-1' down to 1 (K times each)
    #    - 'mid + 1' up to N (K times each) -> sorted descending
    
    # Correcting step 3:
    # After picking the 'low' value, we want the lexicographically largest sequence.
    # This means we pick the largest available numbers first.
    # Available: 
    # - Numbers from {mid + 1, ..., N} (K times each)
    # - Number 'low' (K-1 times)
    # - Numbers from {1, ..., low-1} (K times each)
    # Sorted descending: {N, N-1, ..., mid+1, low, low-1, ..., 1}
    
    # Implementation:
    res = []
    mid = (N + 1) // 2
    low = N // 2
    
    if N % 2 != 0:
        res.extend([mid] * K)
    
    res.append(low)
    
    # Remaining counts
    counts = {i: K for i in range(1, N + 1)}
    # Subtract what we used
    if N % 2 != 0:
        counts[mid] -= K
    counts[low] -= 1
    
    # To get the largest sequence, iterate from N down to 1
    for i in range(N, 0, -1):
        res.extend([i] * counts[i])
        
    print(*(res))

if __name__ == "__main__":
    solve()