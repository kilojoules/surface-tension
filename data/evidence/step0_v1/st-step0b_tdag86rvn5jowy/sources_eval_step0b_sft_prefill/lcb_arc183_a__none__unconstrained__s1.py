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

    # The total number of good sequences S is (NK)! / (K!^N)
    # We want the floor((S+1)/2)-th sequence.
    # Note: S is symmetric. If a sequence is (s1, s2, ..., s_{NK}),
    # its "complement" (relative to the set of available numbers) 
    # is not immediately obvious, but the lexicographical middle 
    # of all permutations of a multiset is reached by 
    # filling the first half with the smallest possible and 
    # the second half with the largest possible, or vice versa.
    
    # Specifically, for a multiset, the S/2-th and (S/2 + 1)-th 
    # sequences are the ones that "split" the space.
    # The floor((S+1)/2)-th sequence is the last sequence that 
    # starts with a number smaller than the "median" element, 
    # or the first sequence that starts with the median if S is odd.
    
    # However, a simpler observation for this specific problem:
    # The set of all good sequences is symmetric. 
    # If we replace each element x with (N + 1 - x), we get another good sequence.
    # This operation reverses the lexicographical order.
    # The floor((S+1)/2)-th sequence is the one immediately preceding 
    # the point where the sequence becomes its own "dual" or 
    # simply the one where we try to pick the smallest available digits 
    # for the first half of the length and largest for the second.
    
    # Actually, the most reliable way to find the middle of a lexicographical 
    # range without calculating S (which is huge) is to realize that 
    # the "middle" sequence is constructed by:
    # For each position i from 1 to NK:
    # We want to pick the digit d such that the number of sequences 
    # starting with the current prefix is just under S/2.
    
    # But we can't compute S. Let's use the symmetry:
    # The floor((S+1)/2)-th sequence is the one where we 
    # effectively "fill" the sequence such that we are as close to 
    # the center as possible.
    # This is achieved by:
    # For the first NK // 2 positions, we want to pick the smallest possible digits.
    # For the remaining positions, we want to pick the largest possible digits.
    # Wait, that's for a different problem.
    
    # Correct logic for the middle of a multiset permutation:
    # The middle sequence is the one where we take the elements of the 
    # multiset, sort them, and arrange them such that the first half 
    # are the smallest and the second half are the largest? No.
    
    # Let's use the property: the floor((S+1)/2)-th sequence is the 
    # lexicographically last sequence that is "smaller than or equal to" 
    # its own dual (where dual is replacing x with N+1-x).
    # A sequence A is <= dual(A) if at the first index i where A_i != dual(A)_i, 
    # A_i < dual(A)_i.
    
    # To find the floor((S+1)/2)-th sequence:
    # For each position i = 0, 1, ..., NK-1:
    # We try digits d = 1, 2, ..., N.
    # If we pick d, we need to check if the number of sequences 
    # starting with the current prefix is <= S/2.
    # This is still hard.
    
    # Let's reconsider: the middle sequence of all permutations of a multiset.
    # If we have the sorted elements of the multiset: L = [1]*K + [2]*K + ... + [N]*K
    # The middle sequence is constructed by:
    # For i = 0 to NK-1:
    # The available digits are sorted. The "middle" choice is the median of the 
    # remaining digits.
    # If we have remaining digits, and we pick the median, we split the 
    # remaining permutations into two equal halves.
    
    # Let the remaining digits be a multiset M.
    # Let the distinct digits be u1 < u2 < ... < um.
    # The number of permutations starting with ui is S_i = (sum(counts)-1)! / product(counts!).
    # S_i = S * (count of ui) / (sum of counts).
    # We want to find the smallest j such that sum_{i=1}^j S_i >= S/2.
    # sum_{i=1}^j S_i = S * (sum_{i=1}^j count of ui) / (sum of counts).
    # So we need (sum_{i=1}^j count of ui) / (sum of counts) >= 1/2.
    # sum_{i=1}^j count of ui >= (sum of counts) / 2.
    
    # This means at each step, we pick the digit u_j that covers the 50% mark 
    # of the remaining elements.
    # Specifically, if we are looking for the k-th permutation where k = S/2:
    # We find j such that sum_{i=1}^{j-1} S_i < k <= sum_{i=1}^j S_i.
    # Then the first digit is u_j, and we look for the (k - sum_{i=1}^{j-1} S_i)-th 
    # permutation of the remaining multiset.
    # The new k becomes k' = k - sum_{i=1}^{j-1} S_i.
    # The ratio k'/S' = (k - sum_{i=1}^{j-1} S_i) / (S * (count of uj / sum of counts))
    # = (k/S - (sum_{i=1}^{j-1} count of ui / sum of counts)) / (count of uj / sum of counts).
    
    # Let r = k/S. Initially r = 1/2.
    # At each step, we find j such that:
    # sum_{i=1}^{j-1} (count of ui / total) < r <= sum_{i=1}^j (count of ui / total).
    # The first digit is u_j.
    # The new r is (r - sum_{i=1}^{j-1} (count of ui / total)) / (count of uj / total).
    
    # To avoid floating point, we use fractions or simply track the 
    # cumulative counts.
    # Let current_total = sum of counts.
    # We seek j such that 2 * (sum_{i=1}^{j-1} count of ui) < 2 * r * current_total <= 2 * (sum_{i=1}^j count of ui).
    # Since r starts at 1/2, 2 * r * current_total is just current_total.
    # We need sum_{i=1}^{j-1} count of ui < (current_total + 1) / 2 <= sum_{i=1}^j count of ui.
    # Wait, the problem asks for floor((S+1)/2).
    # If S is even, it's S/2. If S is odd, it's (S+1)/2.
    # This is exactly the sequence that occupies the middle.
    
    # Let's use the property: the k-th permutation of a multiset.
    # We can maintain the rank as a fraction p/q.
    # Initial rank: p=1, q=2 (since we want the S/2-th).
    # Actually, let's use the property that the "middle" sequence is 
    # simply the one where we pick the median of the remaining elements 
    # and recurse.
    # For a multiset, the "middle" element is the one at index (total-1)//2.
    # If we sort the remaining elements, the element at (total-1)//2 
    # will be the first digit of the floor((S+1)/2)-th sequence.
    # Then we remove it and repeat.
    
    # Let's test this hypothesis with Sample 1: N=2, K=2. 
    # Elements: [1, 1, 2, 2]. Total=4.
    # i=0: index (4-1)//2 = 1. Element at index 1 is 1. Sequence: (1, ...)
    # Remaining: [1, 2, 2]. Total=3.
    # i=1: index (3-1)//2 = 1. Element at index 1 is 2. Sequence: (1, 2, ...)
    # Remaining: [1, 2]. Total=2.
    # i=2: index (2-1)//2 = 0. Element at index 0 is 1. Sequence: (1, 2, 1, ...)
    # Remaining: [2]. Total=1.
    # i=3: index (1-1)//2 = 0. Element at index 0 is 2. Sequence: (1, 2, 1, 2)
    # Sample 1 output is (1, 2, 2, 1). My hypothesis is wrong.
    
    # Correct logic:
    # The number of sequences starting with $u_1, \dots, u_{j-1}$ is $S \times \frac{\sum_{i=1}^{j-1} \text{count}(u_i)}{\text{total}}$.
    # We want the smallest $j$ such that $\sum_{i=1}^j \text{count}(u_i) \geq \frac{\text{total} + 1}{2}$ is NOT the way.
    # The rank is relative. Let the current rank be $R$ (1-indexed) and total permutations be $S$.
    # We pick $u_j$ such that $\sum_{i=1}^{j-1} S_i < R \leq \sum_{i=1}^j S_i$.
    # Then $R_{new} = R - \sum_{i=1}^{j-1} S_i$.
    # $S_{new} = S_j$.
    # $R_{new}/S_{new} = (R - \sum_{i=1}^{j-1} S_i) / S_j$.
    # Let $r = R/S$. Initially $r = 1/2$.
    # $R_{new}/S_{new} = (r - \sum_{i=1}^{j-1} \frac{count(u_i)}{total}) / \frac{count(u_j)}{total}$.
    
    # Let's trace Sample 1: N=2, K=2. S=6. floor((6+1)/2) = 3.
    # R=3, S=6, r=0.5.
    # u1=1, count=2. S1 = 6 * (2/4) = 3.
    # Since R <= S1 (3 <= 3), the first digit is u1 = 1.
    # R_new = 3, S_new = 3, r_new = 3/3 = 1.0.
    # Remaining: {1:1, 2:2}. Total=3.
    # u1=1, count=1. S1 = 3 * (1/3) = 1.
    # Since R > S1 (3 > 1), we move to u2.
    # R_new = 3 - 1 = 2, S_new = 3 * (2/3) = 2, r_new = 2/2 = 1.0.
    # First digit was 1, second is 2.
    # Remaining: {1:1, 2:1}. Total=2.
    # u1=1, count=1. S1 = 2 * (1/2) = 1.
    # Since R > S1 (2 > 1), we move to u2.
    # R_new = 2 - 1 = 1, S_new = 2 * (1/2) = 1, r_new = 1/1 = 1.0.
    # Third digit is 2.
    # Remaining: {1:1}. Total=1.
    # Fourth digit is 1.
    # Result: (1, 2, 2, 1). Correct.
    
    # To implement this without floats:
    # Maintain R and S as a fraction or just use the property:
    # R_new = (R - sum_{i=1}^{j-1} S_i) * S_j / S_