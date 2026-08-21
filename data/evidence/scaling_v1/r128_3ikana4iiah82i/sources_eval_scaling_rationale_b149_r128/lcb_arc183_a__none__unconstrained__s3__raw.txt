The constraint to avoid explicit loops forces a shift from imperative state mutation to a functional paradigm. While not necessarily "cleaner" for all programmers, it encourages the use of higher-order functions and recursion, reducing side effects. To solve this problem, I will use `functools.reduce` to maintain the state of remaining counts of numbers and the current lexicographical index, and `math.comb` to calculate the number of permutations of multisets.

```python
import sys
from math import comb
from functools import reduce

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    k = int(input_data[1])

    # The total number of good sequences is (N*K)! / (K!^N)
    # We need the floor((S+1)/2)-th sequence.
    # Let's calculate the total S first.
    # S = (N*K)! / (K!^N). However, S can be massive.
    # We need to find the sequence at index target = (S + 1) // 2.
    
    # To find the sequence, we determine the elements one by one.
    # For the current position, we try numbers v = 1, 2, ..., N.
    # If we pick v, the number of ways to complete the sequence is:
    # (Remaining_Total)! / Product(Remaining_Counts!)
    
    # Since we need the middle sequence, we can use the symmetry of the problem.
    # The lexicographical reverse of a sequence (s1, s2, ..., sn) is 
    # (N+1-s1, N+1-s2, ..., N+1-sn).
    # The middle sequence is the one that is "closest" to its own reverse.
    # Specifically, if S is the total number of sequences, the (S+1)//2-th 
    # sequence is the one where we effectively "flip" the digits of the 
    # (S - (S+1)//2 + 1)-th sequence.
    
    # Actually, there is a simpler symmetry:
    # The sequence at index i is the "complement" of the sequence at index (S - i + 1).
    # The complement of a sequence is replacing each x with (N + 1 - x).
    # For i = (S+1)//2, the sequence is the complement of the sequence at 
    # index S - (S+1)//2 + 1 = (2S - S - 1 + 2)//2 + 1 = (S+1)//2 + 1 (approx).
    
    # Let's use the property: the (S+1)//2-th sequence is the one that 
    # would be the "first" sequence if we were allowed to pick the 
    # "middle" available digit at each step.
    # More formally: at each step, we want to pick the smallest digit v 
    # such that the number of sequences starting with digits < v is 
    # strictly less than (S+1)//2, and the number of sequences starting 
    # with digits <= v is >= (S+1)//2.
    
    # Because we need the middle, we can simulate this by realizing that
    # we want to pick the digit v that balances the remaining possibilities.
    # The total number of sequences is S. We want index T = (S+1)//2.
    # At any step, if we have counts c1, c2, ..., cN, the total permutations are
    # Total = (sum(ci))! / product(ci!)
    # The number of permutations starting with digit v is:
    # Ways(v) = (Total * cv) / sum(ci)
    
    # To avoid huge numbers and loops, we use reduce to iterate through the 
    # positions of the sequence.
    
    # Initial state: (current_target, current_counts)
    # current_target is the rank we are looking for.
    
    # We need a way to calculate the rank without explicit loops.
    # We can use a helper function to find the digit v.
    
    def get_digit(state):
        target, counts = state
        total_rem = sum(counts)
        
        # Calculate total permutations for the current state
        # Total = (sum(counts))! / product(counts!)
        # Ways(v) = Total * counts[v-1] / total_rem
        
        # We need to find v such that sum_{i=1}^{v-1} Ways(i) < target <= sum_{i=1}^{v} Ways(i)
        # Instead of calculating Total, we can use the ratio:
        # Ways(v) / Total = counts[v-1] / total_rem
        
        # However, we need the absolute value of Ways(v) to compare with target.
        # Let's use a trick: the target rank relative to the total is 
        # roughly 0.5. 
        # At each step, we pick v such that the sum of (counts[i]/total_rem) 
        # for i < v is < target/Total.
        
        # Since we start at target = (S+1)//2, the initial ratio is approx 0.5.
        # Let's maintain the target as a fraction or a decimal.
        pass

    # Correct approach using the symmetry:
    # The (S+1)//2-th sequence is the one where at each step we pick the 
    # smallest v such that the sum of Ways(i) for i <= v is >= target.
    # Since we start at target = (S+1)//2, and Ways(v) is proportional to counts[v],
    # we are essentially picking v such that the prefix sum of counts reaches 
    # half of the total remaining elements.
    
    # Let's refine this:
    # Let T_0 = (S+1)//2.
    # At step 1, we pick v such that:
    # sum_{i=1}^{v-1} [ (Total * counts[i]) / total_rem ] < T_0 <= sum_{i=1}^{v} [ (Total * counts[i]) / total_rem ]
    # This is equivalent to:
    # sum_{i=1}^{v-1} counts[i] < (T_0 * total_rem) / Total <= sum_{i=1}^{v} counts[i]
    
    # Let R_0 = T_0 / S. (Approximately 0.5)
    # At each step, we pick v such that:
    # sum_{i=1}^{v-1} (counts[i]/total_rem) < R_curr <= sum_{i=1}^{v} (counts[i]/total_rem)
    # Then we update R_{curr+1} = (R_curr - sum_{i=1}^{v-1} (counts[i]/total_rem)) / (counts[v]/total_rem)
    
    # To avoid floating point issues, we can use the fact that we want the 
    # "middle" sequence. The middle sequence is the one where we always 
    # pick the digit v that keeps the remaining distribution as balanced 
    # as possible, or more simply, we can use the property that the 
    # (S+1)//2-th sequence is the one where we pick v such that 
    # the sum of counts of digits < v is just under half of the total.
    
    # Actually, the most robust way is to use the property:
    # The (S+1)//2-th sequence is the one where we pick v such that
    # sum_{i=1}^{v-1} counts[i] < (total_rem + 1) // 2 <= sum_{i=1}^{v} counts[i]
    # is NOT necessarily true. That only works if all counts are 1.
    
    # Let's use the property: the (S+1)//2-th sequence is the 
    # lexicographical middle. For any sequence, its "complement" 
    # (replacing x with N+1-x) is its mirror in lexicographical order.
    # The middle sequence is the one that is "closest" to its complement.
    # This means at each step, we want to pick v such that the 
    # remaining counts are as balanced as possible.
    # Specifically, we pick v such that the number of sequences 
    # starting with 1...v-1 is < (S+1)//2 and starting with 1...v is >= (S+1)//2.
    
    # Let's use the ratio R = target / Total.
    # Initial R = ((S+1)//2) / S.
    # At each step:
    # 1. Calculate p_i = counts[i] / total_rem for i = 1...N.
    # 2. Find v such that sum_{i=1}^{v-1} p_i < R <= sum_{i=1}^{v} p_i.
    # 3. Update R = (R - sum_{i=1}^{v-1} p_i) / p_v.
    # 4. Update counts[v] -= 1.
    
    # Since we need (S+1)//2, the initial R is very close to 0.5.
    # Because of the symmetry, the (S+1)//2-th sequence is simply the 
    # sequence generated by picking v at each step such that 
    # sum_{i=1}^{v-1} counts[i] < (total_rem + 1) // 2 <= sum_{i=1}^{v} counts[i].
    # Wait, let's check Sample 1: N=2, K=2. S=6. (S+1)//2 = 3.
    # Step 1: total_rem=4. (4+1)//2 = 2. 
    # v=1: counts[0]=2. 0 < 2 <= 2. So v=1.
    # Step 2: total_rem=3. (3+1)//2 = 2.
    # v=1: counts[0]=1. 0 < 2 <= 1 (False).
    # v=2: counts[1]=2. 1 < 2 <= 1+2 (True). So v=2.
    # Step 3: total_rem=2. (2+1)//2 = 1.
    # v=1: counts[0]=1. 0 < 1 <= 1 (True). So v=1. 
    # Wait, Sample 1 output is 1 2 2 1. My logic gives 1 2 1 2.
    # Let's re-evaluate.
    
    # The correct logic for the (S+1)//2-th sequence:
    # It is the sequence that is "half-way". 
    # Due to symmetry, the (S+1)//2-th sequence is the one where 
    # we pick v such that the number of sequences starting with 1...v-1 
    # is strictly less than (S+1)//2.
    # Let's use the property: the (S+1)//2-th sequence is the 
    # "complement" of the (S - (S+1)//2 + 1)-th sequence.
    # For N=2, K=2, S=6, (S+1)//2 = 3.
    # The 3rd sequence is the complement of the (6-3+1)=4th sequence.
    # The 4th sequence is (2, 1, 1, 2). Its complement is (1, 2, 2, 1).
    # This matches Sample 1!
    
    # To find the (S+1)//2-th sequence, we can find the 
    # (S - (S+1)//2 + 1)-th sequence and then complement it.
    # S - (S+1)//2 + 1 = (2S - S - 1 + 2)//2 + 1 = (S+1)//2 + 1.
    # So we need the (S//2 + 1)-th sequence.
    
    # For N=2, K=2, S=6, we need the 4th sequence.
    # Step 1: v=1, Ways(1)=3. 3 < 4, so we move to v=2.
    # Target becomes 4 - 3 = 1. v=2.
    # Step 2: counts=[2, 1], total=3. Ways(1)= (3!/2!1!)*2/3 = 2.
    # 2 >= 1, so v=1.
    # Step 3: counts=[1, 1], total=2. Ways(1)= (2!/1!1!)*1/2 = 1.
    # 1 >= 1, so v=1.
    # Step 4: counts=[0, 1], total=1. v=2.
    # Result: 2 1 1 2