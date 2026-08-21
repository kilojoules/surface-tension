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
    # This is the "middle" sequence. 
    # Due to the symmetry of lexicographical order, if a sequence T is the i-th,
    # its "complement" (where each value x is replaced by N - x + 1) 
    # is the (S - i + 1)-th sequence.
    # The middle sequence is the one that is "closest" to its own complement.
    # Specifically, for S sequences, the floor((S+1)/2)-th sequence is the 
    # lexicographically largest sequence that is <= its complement.
    
    # To find the floor((S+1)/2)-th sequence, we can use the property that
    # the set of all good sequences is symmetric.
    # The sequence we are looking for is the one that starts with the smallest 
    # possible digit such that at least half of all sequences start with that 
    # digit or a smaller one.
    
    # However, a simpler observation:
    # The floor((S+1)/2)-th sequence is the one that is "just before" the 
    # point where the sequence becomes larger than its complement.
    # This is equivalent to finding the sequence T such that T is the 
    # lexicographically largest sequence where T <= complement(T).
    
    # For a sequence T, T <= complement(T) if at the first index i where 
    # T[i] != complement(T)[i], we have T[i] < complement(T)[i].
    # If T[i] = complement(T)[i] for all i, then T = complement(T).
    
    # The condition T[i] < complement(T)[i] is T[i] < N - T[i] + 1,
    # which simplifies to 2 * T[i] < N + 1.
    
    # To maximize T while keeping T <= complement(T):
    # 1. We want the first index i where T[i] != complement(T)[i] to be as late as possible.
    # 2. At that index i, we want T[i] to be the largest integer such that 
    #    T[i] < N - T[i] + 1.
    # 3. For all j < i, we must have T[j] = complement(T)[j].
    #    This is only possible if N is odd and T[j] = (N+1)/2, or if we can't 
    #    satisfy T[j] = complement(T)[j].
    #    Wait, T[j] = complement(T)[j] means T[j] = N - T[j] + 1, so 2*T[j] = N+1.
    #    This requires N to be odd and T[j] = (N+1)//2.
    
    # Let's refine:
    # We want the largest T such that T <= complement(T).
    # This means at the first index i where T[i] != complement(T)[i], T[i] < complement(T)[i].
    # To make T largest, we want to keep T[i] = complement(T)[i] for as long as possible.
    # T[i] = complement(T)[i] is only possible if N is odd and T[i] = (N+1)//2.
    # We can do this for at most K times.
    
    # Case 1: N is even.
    # T[i] can never equal complement(T)[i].
    # The first index i=0 must satisfy T[0] < complement(T)[0].
    # To maximize T, we pick T[0] = N // 2.
    # Then for all subsequent indices, we want T to be as large as possible.
    # So we fill the remaining slots with the remaining numbers in descending order.
    
    # Case 2: N is odd.
    # T[i] = complement(T)[i] if T[i] = (N+1)//2.
    # We can have T[i] = (N+1)//2 for K times.
    # If we do that for all K occurrences of (N+1)//2, we then need the next 
    # available digit T[i] to be < complement(T)[i].
    # The remaining digits are {1, ..., (N-1)//2} and {(N+3)//2, ..., N}.
    # To maximize T, we want the first "differing" digit to be as large as possible,
    # which is (N-1)//2.
    # Then all subsequent digits should be as large as possible (descending).

    # General Algorithm:
    # 1. Use as many (N+1)//2 as possible (if N is odd) at the start? 
    #    No, that would make the sequence smaller. 
    #    To make T largest, we want the first index i where T[i] != complement(T)[i]
    #    to be as late as possible.
    #    If N is odd, we can have T[i] = (N+1)//2 for K indices.
    #    But we can only do this if those indices are the ONLY indices.
    #    Actually, the most "middle" sequence is simply the one that is 
    #    lexicographically the largest among those T <= complement(T).
    
    # Correct Logic for floor((S+1)/2)-th:
    # It is the sequence T that is the largest such that T <= complement(T).
    # To maximize T:
    # - We want T[i] to be as large as possible.
    # - The constraint is that at the first index i where T[i] != N - T[i] + 1,
    #   we must have T[i] < N - T[i] + 1.
    # - To push this index i as far back as possible, we need T[i] = N - T[i] + 1
    #   for all i < first_diff. This requires N to be odd and T[i] = (N+1)//2.
    # - We can do this for at most K times.
    # - After we run out of (N+1)//2 or we decide to stop, the next digit T[i]
    #   must be < N - T[i] + 1. The largest such digit is (N // 2).
    # - After that, all remaining digits should be placed in descending order to maximize T.

    # Wait, if N is odd, we can place all K copies of (N+1)//2 first?
    # Let's check Sample 4: N=3, K=3. Complement of (2,2,2,1,3,3,3,1,1) 
    # is (2,2,2,3,1,1,1,3,3).
    # (2,2,2,1,3,3,3,1,1) < (2,2,2,3,1,1,1,3,3). Correct.
    # And it's the largest such sequence.
    
    # Construction:
    # 1. If N is odd, start with K copies of (N+1)//2.
    # 2. Then, the next digit must be the largest digit < (N+1)//2, which is (N // 2).
    #    Wait, if N is odd, the digits are 1, ..., (N-1)//2, (N+1)//2, (N+3)//2, ..., N.
    #    The digit (N+1)//2 is its own complement.
    #    To maximize T, we use all K of (N+1)//2, then the next digit must be < its complement.
    #    The available digits are {1... (N-1)//2} and {(N+3)//2 ... N}.
    #    The complement of x is N-x+1.
    #    We want the largest T[i] such that T[i] < N - T[i] + 1.
    #    That is T[i] < (N+1)/2. The largest such integer is (N-1)//2.
    #    Then we fill the rest descending.
    
    # Let's refine:
    # If N is even:
    #   First digit: N // 2
    #   Remaining: All other digits (1...N) K times, minus one N//2, sorted descending.
    # If N is odd:
    #   First K digits: (N+1) // 2
    #   Next digit: (N-1) // 2
    #   Remaining: All other digits (1...N) K times, minus K*(N+1)//2 and one (N-1)//2, sorted descending.

    # Let's trace Sample 1: N=2, K=2.
    # N is even. First digit: 2 // 2 = 1.
    # Remaining: {1, 2, 2} sorted descending: 2, 2, 1.
    # Result: 1 2 2 1. Correct.
    
    # Sample 3: N=6, K=1.
    # N is even. First digit: 6 // 2 = 3.
    # Remaining: {1, 2, 4, 5, 6} sorted descending: 6, 5, 4, 2, 1.
    # Result: 3 6 5 4 2 1. Correct.
    
    # Sample 4: N=3, K=3.
    # N is odd. First K digits: (3+1)//2 = 2, 2, 2.
    # Next digit: (3-1)//2 = 1.
    # Remaining: {1, 1, 3, 3, 3} sorted descending: 3, 3, 3, 1, 1.
    # Result: 2 2 2 1 3 3 3 1 1. Correct.

    # Implementation:
    res = []
    if N % 2 == 0:
        first = N // 2
        res.append(first)
        # Remaining digits
        rem = [i for i in range(1, N + 1) for _ in range(K)]
        rem.remove(first)
        res.extend(sorted(rem, reverse=True))
    else:
        mid = (N + 1) // 2
        res.extend([mid] * K)
        first_diff = (N - 1) // 2
        res.append(first_diff)
        # Remaining digits
        rem = [i for i in range(1, N + 1) for _ in range(K)]
        # Remove all mid and one first_diff
        # Using a list comprehension to remove all mid
        rem = [i for i in rem if i != mid]
        # Remove one first_diff
        # Since we can't use .remove() inside a comprehension, we do it carefully
        # We need to remove exactly one instance of first_diff
        # A trick to remove one instance:
        # Sort rem, find index of first_diff, and slice it out.
        rem.sort()
        # Find index of first_diff
        # Since first_diff is (N-1)//2, it must exist in rem.
        # We can use a generator to find the first index.
        idx = next(i for i, x in enumerate(rem) if x == first_diff)
        rem.pop(idx)
        res.extend(sorted(rem, reverse=True))

    print(*(res))

if __name__ == "__main__":
    solve()