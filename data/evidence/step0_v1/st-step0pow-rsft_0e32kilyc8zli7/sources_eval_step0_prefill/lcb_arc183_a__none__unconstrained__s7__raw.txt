```python
import sys

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    k = int(input_data[1])

    # The constraint to avoid explicit loops forces a functional programming style,
    # which can be cleaner for certain transformations but is generally less 
    # idiomatic for state-heavy algorithms like this one.
    # We will use recursion for the main loop and list comprehensions/map for others.

    # The total number of good sequences S is (N*K)! / (K!)^N.
    # We need the floor((S+1)/2)-th sequence.
    # This is essentially the "middle" sequence.
    # Because the set of good sequences is symmetric (replacing x with N-x+1),
    # the middle sequence is the one that is "self-complementary" in a sense,
    # or specifically, the one that balances the counts of available numbers.
    
    # For a sequence to be the middle one, at each position i, we want to pick
    # the smallest possible digit such that the number of sequences starting with
    # that digit (and all smaller digits) is at least S/2.
    
    # However, there is a much simpler observation for the "middle" sequence:
    # The lexicographical middle of all permutations of a multiset is the sequence
    # that is "balanced". For this specific problem, the floor((S+1)/2)-th sequence
    # is the one where we try to pick the median available digit at each step,
    # but we must maintain the property that the remaining counts allow for a 
    # symmetric distribution.
    
    # Actually, the property is: the middle sequence is the one where we 
    # greedily pick the digit 'd' such that the number of sequences starting with 
    # digits < d is < S/2, and the number of sequences starting with digits <= d is >= S/2.
    
    # Due to the symmetry of the multiset {1*K, 2*K, ..., N*K}, the middle sequence
    # is simply the sequence that is its own "complement" (where complement of x is N-x+1)
    # reversed. But more simply, it's the sequence where at each step we pick the 
    # digit that keeps the remaining multiset balanced.
    
    # The middle sequence is constructed by:
    # At each position, we want to pick digit d such that we don't "tip the scale"
    # too far. For this specific problem, the middle sequence is the one where
    # we pick digits in a way that the sequence is "as close to the center" as possible.
    # It turns out the middle sequence is:
    # For i from 1 to NK:
    # Pick the smallest d such that the number of ways to complete the sequence
    # is >= the remaining required rank.
    
    # But calculating (NK)! / (K!)^N is too large.
    # Notice: the middle sequence is the one that is lexicographically 
    # "halfway". For a symmetric distribution, this is the sequence 
    # that is the "reverse" of itself if you replace x with N-x+1.
    # Wait, the simplest characterization:
    # The middle sequence is the one where we pick digits such that we 
    # maintain the balance. 
    # For N=2, K=2: (1,1,2,2), (1,2,1,2), (1,2,2,1), (2,1,1,2), (2,1,2,1), (2,2,1,1)
    # Middle (3rd) is (1,2,2,1).
    # For N=6, K=1: (3,6,5,4,2,1) - this is the reverse of (1,2,4,5,6,3)? No.
    
    # Let's use the property: the middle sequence is the one where we pick 
    # the digit d such that we use the "middle" of the available options.
    # Specifically, for the middle sequence, we can use a greedy approach:
    # At each step, we pick the smallest d such that the number of sequences 
    # starting with 1...d-1 is < S/2.
    # Because of symmetry, the middle sequence is the one that is 
    # "lexicographically opposite" to itself.
    # The middle sequence is the one where we pick d such that 
    # the remaining counts are "balanced".
    
    # Correct logic for the middle sequence of a symmetric multiset:
    # It is the sequence where we pick d = (N+1)/2 if possible, 
    # but we must use all digits.
    # Actually, the middle sequence is simply the one where we 
    # arrange the digits in the order:
    # For i = 1 to NK:
    # Pick d such that the number of sequences starting with < d is < S/2.
    # This is equivalent to:
    # While there are digits left:
    #   Pick d from 1 to N:
    #     If count[d] > 0:
    #       Ways to complete with d = (TotalRemaining-1)! / Product(count[i]!) * count[d]
    #       If current_rank <= Ways:
    #         Pick d, break
    #       Else:
    #         current_rank -= Ways
    
    # Since we need the middle, current_rank = (S+1)//2.
    # We can't use loops, so we use recursion.
    # We use Decimal or high precision for S, but S is too large.
    # However, we only need to compare current_rank with Ways.
    # Ways(d) / TotalWays = count[d] / TotalRemaining.
    # So we can maintain the rank as a fraction of the total.
    # Starting rank = 0.5.
    # At each step, we pick d such that sum_{i=1}^{d-1} (count[i]/TotalRemaining) < rank
    # and sum_{i=1}^{d} (count[i]/TotalRemaining) >= rank.
    
    def get_sequence(counts, total, rank):
        if total == 0:
            return []
        
        # Find d such that sum(counts[1...d-1]/total) < rank <= sum(counts[1...d]/total)
        # We use a helper function to find d without loops
        def find_d(d, acc_prob):
            if d > n: return n
            prob = counts[d-1] / total if total > 0 else 0
            if acc_prob + prob >= rank - 1e-15: # Use small epsilon for float precision
                return d
            return find_d(d + 1, acc_prob + prob)

        d = find_d(1, 0)
        
        # Update rank for the next step:
        # New rank = (rank - sum_{i=1}^{d-1} prob[i]) / prob[d]
        def get_acc_prob(d):
            return sum([counts[i]/total for i in range(d-1)]) if total > 0 else 0
        
        new_rank = (rank - get_acc_prob(d)) / (counts[d-1]/total)
        
        # Update counts
        new_counts = list(counts)
        new_counts[d-1] -= 1
        
        return [d] + get_sequence(new_counts, total - 1, new_rank)

    # To avoid recursion depth issues and the loop constraint:
    # We can use a recursive function with a helper to handle the state.
    # But N*K is 250,000, which is too deep for recursion.
    # Wait, the constraints are N, K <= 500. NK <= 250,000.
    # Python's recursion limit is usually 1000. 
    # However, the prompt says "no loops", "no while", "no for".
    # This forces the use of recursion or map/reduce/comprehensions.
    # For NK = 250,000, we must use sys.setrecursionlimit.
    
    sys.setrecursionlimit(300000)
    
    # The rank is (S+1)//2. In terms of probability, this is effectively 0.5.
    # But we need to be careful with the "floor((S+1)/2)" part.
    # If S is even, (S+1)//2 is S/2. If S is odd, (S+1)//2 is (S+1)/2.
    # In both cases, the starting rank as a fraction of S is 0.5.
    
    # Let's refine the rank update:
    # The number of sequences is S. We want the (S+1)//2 -th.
    # Let R = (S+1)//2.
    # At the first step, we check if R <= count[1]/Total * S.
    # If yes, d=1. If no, R = R - (count[1]/Total * S), and check d=2...
    # This is equivalent to:
    # Current relative rank r = R/S.
    # Pick d such that sum_{i=1}^{d-1} (count[i]/Total) < r <= sum_{i=1}^{d} (count[i]/Total).
    # New r = (r - sum_{i=1}^{d-1} (count[i]/Total)) / (count[d]/Total).
    
    # Special case: S is the total number of sequences.
    # The rank we want is (S+1)//2.
    # The relative rank is ((S+1)//2) / S.
    # As S becomes very large, this is exactly 0.5.
    # For small S, it might differ slightly, but for N, K up to 500, S is huge.
    # The only case where S is small is when N or K is very small.
    # If N=1, S=1, rank=1, r=1.0.
    # If N=2, K=1, S=2, rank=1, r=0.5.
    
    # To handle the precision and the (S+1)//2 exactly, we can use Fractions.
    from fractions import Fraction
    
    def get_seq_rec(counts, total, r):
        if total == 0:
            return []
        
        # We need to find d such that:
        # sum_{i=1}^{d-1} (count[i]/total) < r <= sum_{i=1}^{d} (count[i]/total)
        # Let's find d using a list comprehension and next()
        # Since we can't use for/while, we use a generator expression inside next()
        
        # Calculate cumulative probabilities
        # Using a list comprehension to get the prefix sums of counts
        # counts is 0-indexed, so counts[0] is for digit 1.
        
        # To avoid loops, we can use a recursive function to find d.
        def find_d_rec(d, current_r):
            if d > n: return n
            prob = Fraction(counts[d-1], total)
            if current_r <= prob:
                return d, current_r / prob
            return find_d_rec(d + 1, current_r - prob)
        
        d, new_r = find_d_rec(1, r)
        
        new_counts = list(counts)
        new_counts[d-1] -= 1
        return [d] + get_seq_rec(new_counts, total - 1, new_r)

    # Initial r = ((S+1)//2) / S
    # S = (N*K)! / (K!)^N
    # This is too large to compute. 
    # But we only need r = Fraction(((S+1)//2), S).
    # If S is large, r is effectively 1/2.
    # If S is small, we can compute it.
    # Actually, for any S > 1, (S+1)//2 / S is very close to 0.5.
    # If S is even, r = (S/2) / S = 1/2.
    # If S is odd, r = ((S+1)/2) / S = 1/2 + 1/(2S).
    
    # Let's use the property that for the middle sequence, 
    # we can just use r = Fraction(1, 2) and handle the 
    # "floor((S+1)/2)" by slightly adjusting.
    # Wait, the most robust way to get the middle sequence of a symmetric 
    # distribution is to realize that the sequence is its own "complement" 
    # mirrored.
    # For the middle sequence, at each