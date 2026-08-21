```python
import sys

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    k = int(input_data[1])

    # The total number of good sequences S is (NK)! / (K!)^N.
    # We need the floor((S+1)/2)-th sequence.
    # This is exactly the middle sequence if S is even, 
    # or the middle one if S is odd.
    # In a set of permutations of a multiset, the lexicographical 
    # complement of a sequence (S_1, ..., S_{NK}) is 
    # (N+1-S_1, ..., N+1-S_{NK}).
    # The map f(S) = complement(S) is a bijection that reverses 
    # lexicographical order.
    # The "middle" sequence is the one that is its own complement 
    # if such a thing exists, or the one just before the midpoint.
    # Specifically, the sequence we are looking for is the one 
    # where we try to place the 'middle' available number at each position.
    
    # For a multiset, the sequence at index (S+1)//2 is the one that 
    # remains invariant under the complement operation if we 
    # consider the symmetry of the distribution.
    # The sequence is constructed by:
    # For each position i from 1 to NK:
    # We want the smallest x such that the number of sequences 
    # starting with (prefix + x) is enough to reach the target index.
    # However, since we need the exact middle, we can use the property:
    # The middle sequence is the one where at each step we pick the 
    # value x such that the number of sequences starting with values 
    # < x is just under half of the total remaining sequences.
    
    # Due to the symmetry of the multiset {1*K, 2*K, ..., N*K},
    # the sequence at index (S+1)//2 is simply the sequence 
    # constructed by picking the 'median' available number.
    # If N is even, the middle two values are N//2 and N//2 + 1.
    # If N is odd, the middle value is (N+1)//2.
    
    # The symmetry implies that the sequence is:
    # For N=2, K=2: (1, 2, 2, 1)
    # For N=6, K=1: (3, 6, 5, 4, 2, 1)
    # The pattern is: 
    # For each pair (i, N-i+1), we place them in a way that balances.
    # Actually, the middle sequence of a symmetric multiset is 
    # the one that is "self-complementary" in terms of rank.
    # The sequence is: 
    # For i from 1 to N:
    # If i < (N+1)/2, we have i and N-i+1.
    # The middle sequence is constructed by placing 
    # (N//2) K times, then (N//2 + 1) K times... 
    # Wait, the sample 3 (6, 1) gives 3 6 5 4 2 1.
    # This is: 3, (6, 5, 4, 2, 1). 
    # Sample 4 (3, 3) gives 2 2 2 1 3 3 3 1 1.
    # This is: 2(K times), 1(K times), 3(K times), 1(K times)... no.
    # Let's re-examine: 2 2 2 1 3 3 3 1 1.
    # That is: 2 repeated 3 times, then 1 repeated 1 time, 
    # then 3 repeated 3 times, then 1 repeated 2 times.
    # Actually, the pattern is:
    # Place (N+1)//2 K times.
    # Then place the remaining numbers in pairs (i, N-i+1) 
    # from the middle outwards.
    # For N=3, K=3: Middle is 2. 
    # Sequence: 2,2,2, 1,3,3,3,1,1 (Wait, this is not symmetric).
    # Let's look at the sample 4 again: 2 2 2 1 3 3 3 1 1.
    # This is: 2(K), 1(1), 3(K), 1(K-1).
    # For N=6, K=1: 3, 6, 5, 4, 2, 1.
    # This is: 3(1), 6(1), 5(1), 4(1), 2(1), 1(1).
    # The logic is:
    # 1. Place (N+1)//2 K times.
    # 2. For i from (N//2) down to 1:
    #    Place i 1 time, then place (N-i+1) K times, then place i (K-1) times.
    # Let's check N=6, K=1:
    # (N+1)//2 = 3. Place 3 (1 time).
    # i=3: (already handled by (N+1)//2 if N was odd, but N=6 is even).
    # If N is even, the "middle" is between N//2 and N//2 + 1.
    # The sample 6 1 -> 3 6 5 4 2 1.
    # Here N=6, K=1. Middle values are 3 and 4.
    # It starts with 3, then 6, 5, 4, 2, 1.
    # This looks like: 3, (6, 5, 4), (2, 1).
    # Let's try this logic:
    # While there are numbers left:
    # Pick the smallest available number 'm' such that 
    # the number of sequences starting with < m is < (S+1)//2.
    # Since we cannot compute S, we use the property that the 
    # middle sequence is the one that is "lexicographically 
    # opposite" to itself.
    # The sequence is:
    # For i from 1 to N:
    # If i == (N+1)//2: print i K times.
    # Then for i from (N//2) down to 1:
    #   print i 1 time, print (N-i+1) K times, print i (K-1) times.
    # Wait, if N is even, (N+1)//2 is 3. 3 is printed K times.
    # Then i=3 is already used? No, i goes from N//2 (3) down to 1.
    # For N=6, K=1:
    # (N+1)//2 = 3. Print 3 (1 time).
    # i=3: 3 is already printed? No, the loop is for i < (N+1)//2.
    # Let's refine:
    # 1. Mid = (N + 1) // 2
    # 2. Print Mid K times.
    # 3. For i from Mid - 1 down to 1:
    #    Print i 1 time, Print (N - i + 1) K times, Print i (K - 1) times.
    # 4. If N is even, we have the value (N // 2 + 1) left.
    #    But the loop above handles (N - i + 1).
    #    For N=6, Mid=3. i=2: 2(1), 5(1), 2(0). i=1: 1(1), 6(1), 1(0).
    #    This gives 3, 2, 5, 1, 6. Missing 4.
    # Correct logic for middle sequence of symmetric multiset:
    # The sequence is S_i = (N + 1) - S_{NK - i + 1}.
    # For the middle one, we want the smallest sequence such that 
    # it is lexicographically $\ge$ its complement.
    # This means at the first index $i$ where $S_i \neq (N+1) - S_i$,
    # we must have $S_i > (N+1) - S_i$.
    # To make it the smallest such sequence, we want $S_i$ to be 
    # as small as possible, so $S_i = \lfloor (N+1)/2 \rfloor + 1$.
    # But we can have $S_i = (N+1) - S_i$ for a while.
    # $S_i = (N+1) - S_i$ only if $S_i = (N+1)/2$.
    # So:
    # 1. Place (N+1)/2 as many times as possible (K times).
    # 2. Then place the smallest possible value that is $> (N+1)/2$.
    #    That is $\lfloor (N+1)/2 \rfloor + 1$.
    # 3. Then to keep the sequence smallest, we fill the rest 
    #    with the remaining numbers in increasing order, 
    #    but we must maintain the symmetry $S_i = (N+1) - S_{NK-i+1}$.
    # Actually, the simplest way to describe the middle sequence is:
    # It is the sequence that starts with (N+1)//2 repeated K times,
    # then for i from 1 to N//2:
    # the remaining slots are filled by i and (N-i+1).
    # To be the smallest sequence $\ge$ its complement:
    # At the first index where we can't put (N+1)//2,
    # we must put the smallest available value $x$ such that $x > (N+1) - x$.
    # That is $x = \lfloor N/2 \rfloor + 1$.
    # Then we fill the remaining positions to be as small as possible,
    # while ensuring the total sequence is $\ge$ its complement.
    # The most lexicographically small sequence that is $\ge$ its complement
    # is:
    # 1. (N+1)//2 repeated K times (if N is odd).
    # 2. Then the smallest available value $x$ such that $x > (N+1)/2$.
    # 3. Then all other remaining values in increasing order.
    # 4. Then the complement of the prefix.
    # Wait, the sample 4: 2 2 2 1 3 3 3 1 1.
    # N=3, K=3. (N+1)//2 = 2.
    # 2 2 2 (K times), then 1 (smallest available), then 3 3 3 (remaining), then 1 1 (remaining).
    # This matches! 2 2 2, 1, 3 3 3, 1 1.
    # Sample 3: 6 1. (N+1)//2 = 3.
    # 3 (K times), then 4 (smallest > 3), then 2 1 (remaining increasing? No, 6 5 4 2 1).
    # Let's use the property: The middle sequence is the one that is 
    # lexicographically smallest among those $S$ where $S \ge \text{complement}(S)$.
    # $S \ge \text{complement}(S)$ means at the first index $i$ where $S_i \neq \text{comp}(S)_i$,
    # we have $S_i > \text{comp}(S)_i$.
    # To minimize $S$, we want $S_i = \text{comp}(S)_i$ for as many $i$ as possible.
    # $S_i = \text{comp}(S)_i \iff S_i = (N+1)/2$.
    # So we put $(N+1)/2$ for $K$ times (if $N$ is odd).
    # Then we need $S_{K+1} > \text{comp}(S)_{K+1}$.
    # $\text{comp}(S)_{K+1} = (N+1) - S_{NK - (K+1) + 1} = (N+1) - S_{NK-K}$.
    # This is getting complex. Let's use the property:
    # The middle sequence is simply the one where we place 
    # (N+1)//2 K times, then 1 K times, then (N+1) K times, 
    # then 2 K times, then N K times... 
    # No, that's