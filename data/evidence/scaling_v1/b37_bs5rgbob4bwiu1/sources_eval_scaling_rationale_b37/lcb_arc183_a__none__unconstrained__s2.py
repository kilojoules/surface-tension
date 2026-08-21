The constraint to avoid explicit loops forces a shift from imperative state management to a functional paradigm. While not necessarily "cleaner" for complex state tracking, it encourages the use of declarative constructs that describe *what* the result is rather than *how* to step through the process. For this problem, I will use `functools.reduce` to simulate the iterative process of building the sequence and `itertools.accumulate` or recursion to handle the counting logic, though `reduce` is the most robust way to maintain the state of remaining counts and the target index across the sequence length.

```python
import sys
from math import factorial
from functools import reduce

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    k = int(input_data[1])

    # The total number of good sequences S is (NK)! / (K!)^N
    # We need the floor((S+1)/2)-th sequence.
    # Let target = (S + 1) // 2
    
    # To avoid calculating S explicitly (which is huge), 
    # we use the property that the middle sequence is the 
    # "complement" of the sequence that would be at index (S+1)//2
    # from the end. Actually, a simpler way to find the middle 
    # is to realize that the sequence at index (S+1)//2 is the 
    # lexicographical "median".
    # The lexicographical opposite of a sequence (s1, s2, ..., sn)
    # is (N+1-s1, N+1-s2, ..., N+1-sn).
    # If S is the total number of sequences, the sequence at 
    # position i and the sequence at position S - i + 1 are complements.
    # The middle sequence (S+1)//2 is its own complement if S is odd,
    # or the smaller of the two middle ones if S is even.
    # Wait, the simplest approach: the target index is (S+1)//2.
    # We can determine the sequence digit by digit.
    
    # Precompute factorials for multinomial coefficients
    # Since N, K <= 500, NK <= 250,000. Factorials are too large.
    # However, we only need to compare the target index with the number of 
    # sequences starting with a certain digit.
    # Let count(r1, r2, ..., rN) be the number of ways to arrange the remaining digits.
    # count = (sum(ri))! / (r1! * r2! * ... * rN!)
    
    # Because we need the (S+1)//2-th sequence, and S can be enormous,
    # we cannot calculate S. But we can use the symmetry:
    # The sequence at index (S+1)//2 is the one where we try to pick the 
    # "middle" available digit at each step.
    # Specifically, for the first position, there are N choices.
    # The number of sequences starting with 1 is S/N.
    # The number of sequences starting with 1 or 2 is 2S/N, etc.
    # The target index (S+1)//2 falls into the range of digit 'd' 
    # if (d-1)*S/N < (S+1)//2 <= d*S/N.
    # This simplifies to: d is the smallest integer such that d/N >= 1/2,
    # which is d = ceil(N/2).
    # But this is only for the first digit. We must maintain this for all positions.
    
    # Correct logic for the median sequence:
    # The median sequence is the one that, at each step, picks the smallest 
    # digit d such that the number of sequences starting with digits < d 
    # is strictly less than (S_current + 1) // 2.
    # Due to symmetry, the median sequence is simply the sequence 
    # constructed by picking the "middle" available digit at each step,
    # but we must be careful with the remaining counts.
    # Actually, the median sequence is the one where we replace each 
    # element x of the lexicographically first sequence with (N + 1 - x)
    # only for the second half of the total sequences.
    # The most direct way: the median sequence is the one where we 
    # always pick the digit d = (N + 1) // 2 if available, 
    # but we must balance the counts.
    
    # Let's use the property: the sequence at (S+1)//2 is the one 
    # that is "lexicographically central".
    # This means at each step, we want to pick digit d such that the 
    # number of ways to complete the sequence using digits < d 
    # is < (S_current + 1) // 2, and using digits <= d is >= (S_current + 1) // 2.
    # Since we are looking for the middle, we can maintain a target 
    # rank and compare it to the multinomial coefficient.
    # To avoid huge numbers, we can use the fact that we want the 
    # middle sequence. The middle sequence is the one where we 
    # effectively "split the difference" of the available digits.
    
    # For the median sequence, at each step, we pick the smallest d 
    # such that the number of sequences starting with 1...d-1 
    # is < (S_current + 1) / 2.
    # This is equivalent to picking d such that the number of sequences 
    # starting with 1...d-1 is less than the number of sequences 
    # starting with d+1...N.
    
    # Let f(r1, ..., rN) be the number of ways to arrange remaining digits.
    # We pick d such that:
    # sum_{i=1}^{d-1} f(r1, ..., ri-1, ..., rN) < (S_curr + 1) / 2 <= sum_{i=1}^{d} f(...)
    # This is equivalent to:
    # sum_{i=1}^{d-1} f(...) < sum_{i=d+1}^{N} f(...) + (f(r_d) + 1) / 2
    
    # Since f(r_i) = (Sum r_j)! / (r1! ... r_i! ... rN!)
    # f(r_i) / f(r_j) = r_i / r_j.
    # So we compare sum_{i=1}^{d-1} r_i with sum_{i=d+1}^{N} r_i.
    
    # Let's trace: we want the smallest d such that 
    # (count of digits < d) < (count of digits > d) + (count of digit d + 1) // 2
    # Wait, the condition is simpler:
    # We pick d such that the number of sequences starting with 1...d-1 
    # is strictly less than the number of sequences starting with d+1...N,
    # OR they are equal and we are at the midpoint of the sequences starting with d.
    
    # Let R = sum of remaining counts.
    # Total sequences S_curr = R! / (r1! ... rN!)
    # Sequences starting with i: S_i = S_curr * (ri / R)
    # We want smallest d such that sum_{i=1}^{d-1} S_i < (S_curr + 1) / 2
    # sum_{i=1}^{d-1} (ri/R * S_curr) < (S_curr + 1) / 2
    # (sum_{i=1}^{d-1} ri) / R < 1/2 + 1/(2 * S_curr)
    # For large S_curr, this is sum_{i=1}^{d-1} ri < R/2.
    
    # Let's refine:
    # At each step, we seek the smallest d such that:
    # (Sum of r_i for i < d) * S_curr / R < (S_curr + 1) / 2
    # And (Sum of r_i for i <= d) * S_curr / R >= (S_curr + 1) / 2
    
    # This simplifies to:
    # 2 * (Sum_{i=1}^{d-1} r_i) * S_curr < R * S_curr + R
    # 2 * (Sum_{i=1}^{d-1} r_i) < R + R/S_curr
    # Since R/S_curr is very small (unless S_curr is small), 
    # this is basically Sum_{i=1}^{d-1} r_i < R/2.
    
    # Let's use a state (current_counts, current_rank_is_above_half)
    # where current_rank_is_above_half is a boolean.
    # If we are exactly at the middle of the total set, rank_above = False.
    # When we pick digit d:
    # Ways to the left: L = sum_{i=1}^{d-1} r_i * (S_curr / R)
    # Ways to the right: Right = sum_{i=d+1}^{N} r_i * (S_curr / R)
    # Ways for current digit: Mid = r_d * (S_curr / R)
    # If L < (S_curr + 1)/2 <= L + Mid, we pick d.
    # New rank within Mid: new_rank = (S_curr + 1)//2 - L
    # New rank_above_half = (new_rank > (Mid + 1) // 2)
    
    # Actually, the most robust way to handle "(S+1)//2" is to track 
    # if we are looking for the "lower" or "upper" median.
    # For a set of size S, the (S+1)//2-th element is the lower median.
    # If we split S into L, Mid, R:
    # 1. If (S+1)//2 <= L: target is in L, new_target = (S+1)//2
    # 2. If L < (S+1)//2 <= L + Mid: target is in Mid, new_target = (S+1)//2 - L
    # 3. If L + Mid < (S+1)//2: target is in R, new_target = (S+1)//2 - L - Mid
    
    # To avoid large numbers, we use the property:
    # L/S = (sum_{i=1}^{d-1} r_i) / R
    # (S+1)//2 / S is approx 1/2.
    # So we compare (sum r_i for i < d) with R/2.
    
    # Let's use a helper function to determine d and the new "relative" target.
    # We can represent the target rank as a fraction or a boolean (above/below middle).
    # Let's use a boolean `above` and a value `rem` to track the position.
    # But the simplest is: the target is the (S+1)//2-th.
    # This is the sequence that is "lexicographically middle".
    # The middle sequence is the one where at each step we pick d such that
    # sum_{i=1}^{d-1} r_i < (R+1)/2 <= sum_{i=1}^{d} r_i
    # Wait, that's only if all r_i were 1. With r_i > 1, it's:
    # sum_{i=1}^{d-1} r_i < R/2 < sum_{i=1}^{d} r_i is not quite right.
    # The correct logic:
    # We want the smallest d such that sum_{i=1}^{d-1} S_i < (S_curr + 1) / 2.
    # S_i = S_curr * (r_i / R).
    # So sum_{i=1}^{d-1} (r_i / R) < 1/2 + 1/(2 * S_curr).
    # This means 2 * (sum_{i=1}^{d-1} r_i) < R + R/S_curr.
    # Since R/S_curr is very small, this is 2 * sum_{i=1}^{d-1} r_i <= R.
    # If 2 * sum_{i=1}^{d-1} r_i < R, then d is the one.
    # If 2 * sum_{i=1}^{d-1} r_i == R, then we must check the +1/2 part.
    
    # Let's use a state: (counts, is_above_mid)
    # is_above_mid: True if we are looking for the (S+1)//2 + 1 -th element, 