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
    # Note: S is symmetric. If we have a sequence (a1, a2, ..., aM),
    # its "complement" (N+1-a1, N+1-a2, ..., N+1-aM) is also a good sequence.
    # Lexicographically, if A < B, then complement(A) > complement(B).
    # The middle two sequences (if S is even) are the (S/2)-th and (S/2 + 1)-th.
    # The floor((S+1)/2)-th sequence is exactly the "middle" of the lexicographical list.
    # Due to the symmetry of the set of all good sequences, the sequence at index (S+1)//2
    # is the one that is "self-complementary" in a sense, or specifically,
    # the one where we effectively pick the "middle" option at each step.
    
    # Let's define the target rank R = (S + 1) // 2.
    # Since we cannot compute S directly (it's too large), we use the property:
    # For the first position, we have N choices.
    # The number of sequences starting with 1 is the same as the number of sequences starting with N.
    # The number of sequences starting with 2 is the same as those starting with N-1, and so on.
    
    # Let f(counts) be the number of ways to arrange the remaining elements.
    # f(counts) = (sum(counts))! / product(counts!)
    # We want to find the R-th sequence.
    # For the first element x in {1, ..., N}:
    # If R <= f(counts - e_x), then the first element is x.
    # If R > f(counts - e_x), then R = R - f(counts - e_x) and move to x+1.
    
    # However, we know R = (S + 1) // 2.
    # S = N * f(counts - e_1) (since all f(counts - e_x) are equal initially).
    # R = (N * f(counts - e_1) + 1) // 2.
    
    # Let g(counts) be the number of sequences that can be formed with the given counts.
    # To find the R-th sequence:
    # For each position i from 1 to NK:
    #   For each candidate x from 1 to N:
    #     if count[x] > 0:
    #       num_ways = g(counts - e_x)
    #       if R <= num_ways:
    #         result[i] = x
    #         counts[x] -= 1
    #         break
    #       else:
    #         R -= num_ways
    
    # To avoid huge numbers, we can observe that we only need to compare R with num_ways.
    # But R changes. We can maintain R as a fraction or use the symmetry.
    # The symmetry property: The (S+1)//2-th sequence is the one where we 
    # effectively treat the "middle" of the distribution.
    
    # Let's use the property: the sequence is the "median" sequence.
    # For the first element:
    # If N is odd, the middle element is (N+1)//2.
    # If N is even, the middle elements are N//2 and N//2 + 1.
    # Since we want floor((S+1)/2), for N=2, K=2, S=6, R=3.
    # f(1,2) = 3!/2!1! = 3.
    # R=3 <= 3, so first element is 1. Remaining counts: {1:1, 2:2}, R=3.
    # Next element: x=1: f(0,2)=1. R=3 > 1, so R=3-1=2.
    # x=2: f(1,1)=2. R=2 <= 2, so second element is 2. Remaining counts: {1:1, 2:1}, R=2.
    # Next element: x=1: f(0,1)=1. R=2 > 1, so R=2-1=1.
    # x=2: f(1,0)=1. R=1 <= 1, so third element is 2. Remaining counts: {1:1, 2:0}, R=1.
    # Last element: 1.
    # Result: 1 2 2 1. Correct for Sample 1.
    
    # To handle the large numbers, we can use the fact that we only need to 
    # compare R with num_ways. We can represent R as a fraction of the total remaining.
    # Let current_rank_ratio = R / g(counts).
    # For x = 1 to N:
    #   prob = count[x] / sum(counts)
    #   if current_rank_ratio <= prob:
    #     element = x
    #     current_rank_ratio = (current_rank_ratio * sum(counts)) / count[x]
    #     break
    #   else:
    #     current_rank_ratio -= prob
    
    # Initial R = (S+1)//2. Initial g(counts) = S.
    # Initial ratio = ((S+1)//2) / S. This is slightly more than 0.5 if S is odd, 0.5 if S is even.
    # Let's use a float or Decimal, but constraints on N, K are 500, so NK=250,000.
    # Precision might be an issue. Let's use the property that we want the "middle" sequence.
    # The middle sequence is the one where we balance the counts.
    # Actually, the most reliable way without loops/recursion is to realize that
    # the (S+1)//2-th sequence is the one where we pick x such that we stay closest to the median.
    # But we can just simulate the process using the ratio.
    
    # Given the constraints and the "middle" requirement, the sequence is:
    # For each position, pick x such that the number of sequences starting with 1...x-1 
    # is < (S+1)//2 and the number of sequences starting with 1...x is >= (S+1)//2.
    # This is equivalent to: pick x such that sum_{j=1}^{x-1} (count[j]/total) < current_ratio <= sum_{j=1}^{x} (count[j]/total).
    
    # Since we can't use loops, we use map/list comprehensions.
    # We need to maintain state across the sequence generation.
    # We can use a mutable object to keep track of counts and the current ratio.
    
    state = {
        'counts': [K] * N,
        'ratio': 0.5, # (S+1)//2 / S is approx 0.5
        'total': N * K
    }
    
    def get_next_element():
        counts = state['counts']
        ratio = state['ratio']
        total = state['total']
        
        # We need to find x such that sum_{j=0}^{x-1} (counts[j]/total) < ratio
        # Let's precompute the cumulative probabilities.
        # Since we can't use loops, we use a trick with a helper function or comprehension.
        
        # To avoid loops, we can use a binary search or a linear scan via a comprehension.
        # But we need to update the state.
        
        # Let's find the smallest x such that sum(counts[0:x]) >= ratio * total
        # We can use a list comprehension to find all x that satisfy this, and take the min.
        target = ratio * total
        # We use a small epsilon to handle floating point issues, 
        # but since we want floor((S+1)/2), we should be careful.
        # Actually, (S+1)//2 / S is exactly 0.5 if S is even, and (S+1)/(2S) if S is odd.
        # (S+1)/(2S) = 0.5 + 1/(2S).
        
        # To avoid floats, we can use the property:
        # The middle sequence is the one where we pick x such that 
        # the number of elements smaller than x is roughly equal to the number of elements larger than x.
        # Specifically, for the first element, if N is even, we have N/2 elements <= N/2 and N/2 elements > N/2.
        # The first half of sequences start with 1...N/2, the second half with N/2+1...N.
        # The (S+1)//2-th sequence will start with N/2 if S is even? No.
        # Let's use the ratio 0.5 and a very small offset.
        
        # Correct logic:
        # For a position, we have counts [c1, c2, ..., cN]. Total T = sum(counts).
        # We pick x if sum_{j=1}^{x-1} (cj * g(counts-ej)) < R <= sum_{j=1}^{x} (cj * g(counts-ej))
        # Note: g(counts-ej) = (T-1)! / product(ck!) * cj = (T-1)! / product(ck!) * cj.
        # All g(counts-ej) for the same j are the same? No, they depend on cj.
        # The number of sequences starting with x is (T-1)! / (c1! ... (cx-1)! ... cN!).
        # This is (T-1)! / (product c_k!) * cx.
        # The fraction of sequences starting with x is cx / T.
        
        # So we pick x such that sum_{j=1}^{x-1} (cj/T) < R/S <= sum_{j=1}^{x} (cj/T).
        # Let current_ratio = R/S.
        # For the first element, current_ratio = ((S+1)//2) / S.
        # If we pick x, the new ratio is (R - sum_{j=1}^{x-1} (cj/T)*S) / ( (cj/T)*S )
        # = (current_ratio - sum_{j=1}^{x-1} (cj/T)) / (cj/T)
        
        # To avoid loops, we can use a recursive-like structure with map or a custom reducer.
        # But we can just use a list comprehension to find x and then update.
        
        # Since we can't use while/for, we use a recursive function.
        # Python's recursion limit is 1000, but NK is 250,000.
        # We must use an iterative approach. The prompt says "no for or while loops".
        # We can use `map` or `list comprehensions` or `reduce`.
        
        return None

    # Let's use the symmetry property:
    # The (S+1)//2-th sequence is the one where we always pick x such that 
    # the remaining counts are as balanced as possible.
    # Actually, there is a simpler way:
    # The sequence is the "middle" one. For each position, we want to pick x such that
    # the number of sequences starting with 1...x-1 is just under 50% of the total.
    # This means we pick x such that sum_{j=1}^{x-1} count[j] < (total + 1) / 2 <= sum_{j=1}^{x} count[j].
    # Wait, this is only true if all remaining counts are equal.
    # If counts are not equal, the "middle" sequence is the one where we 
    # effectively treat the sequence as a string and find the median.
    # The median of all permutations of a multiset is the one that is 
    # lexicographically central.
    # This is achieved by: for each position, pick x such that 
    # sum_{j=1}^{x-1} (count[j] * g(counts-ej)) < S/2 <= sum_{j=1}^{x} (count[j] * g(counts-ej)).
    # Since g(counts-ej) = g(counts) * (count[j]/total),
    # this is sum_{j=1}^{x-1} (count[j]/total) < 1/2 <= sum_{j=1}^{x} (