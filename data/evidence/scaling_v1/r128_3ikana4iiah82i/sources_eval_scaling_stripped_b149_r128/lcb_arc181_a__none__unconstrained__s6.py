import sys
from itertools import groupby

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to process T test cases
    # We use next(input_data) to consume T, then a loop to process each case
    # Since we cannot use a loop, we use a recursive-like structure via map/lambda
    # However, the constraint allows us to use a list comprehension to iterate T times
    
    # To handle the input stream without a loop, we capture the iterator in a list
    # and use a helper function to extract the N and the permutation.
    
    def process_cases(it):
        try:
            # Extract T
            t_val = next(it)
            # For each test case, we need to extract N and then N elements.
            # We use a list comprehension to drive the process.
            # Since we need to track the state of the iterator, we use a helper.
            return [get_result(it) for _ in range(t_val)]
        except StopIteration:
            return []

    def get_result(it):
        # Extract N
        n = next(it)
        # Extract the permutation P
        p = [next(it) for _ in range(n)]
        
        # The problem asks for the minimum operations to sort P.
        # An operation k sorts [1, k-1] and [k+1, N].
        # If P is already sorted, answer is 0.
        # If there exists k such that sorting [1, k-1] and [k+1, N] sorts the whole array,
        # it means P_k must be the value that belongs at position k (which is k),
        # and the set of values {P_1...P_{k-1}} must be {1...k-1} 
        # and {P_{k+1}...P_N} must be {k+1...N}.
        # Actually, the condition is simpler: 
        # If we pick k, the resulting array is sorted if and only if:
        # 1. P_k = k
        # 2. The elements in positions 1 to k-1 are some permutation of 1 to k-1
        # 3. The elements in positions k+1 to N are some permutation of k+1 to N
        # Condition 2 is equivalent to: max(P_1...P_{k-1}) = k-1
        # Condition 3 is equivalent to: min(P_{k+1}...P_N) = k+1
        
        # However, we can simplify: 
        # If P is not sorted, we can always sort it in 2 operations:
        # Op 1: k=1 (sorts 2...N), Op 2: k=N (sorts 1...N-1).
        # Wait, the sample says for (3,2,1,7,5,6,4), k=4 then k=3 works.
        # Let's re-evaluate:
        # If P is already sorted: 0.
        # If there exists k such that P_k = k AND 
        # (all P_i < k for i < k) AND (all P_i > k for i > k): 1.
        # Otherwise: 2.
        
        # To check the "1" condition efficiently:
        # We need P_k = k and max(P_1...P_{k-1}) < k and min(P_{k+1}...P_N) > k.
        # This is equivalent to saying that the set {P_1...P_k} is {1...k} 
        # AND the set {P_k...P_N} is {k...N}.
        
        # Let's use a different approach:
        # The only way it takes 1 operation is if there is some k where
        # sorting the prefix and suffix independently fixes the array.
        # This happens if P_k = k and the elements are already partitioned.
        # If P is not sorted, can it always be done in 2?
        # Yes: k=1 sorts P[2...N], then k=N sorts P[1...N-1].
        # Since P[2...N] is sorted, P[1] is the only one out of place.
        # Then sorting P[1...N-1] will put the original P[1] (which is now at P[2])
        # and all others in the correct place.
        # Actually, the simplest 2-step is:
        # 1. k=1: P becomes (P_1, 1, 2, ..., N-1) if P_1 was N.
        # Wait, if we pick k=1, we sort P[2...N].
        # If we then pick k=N, we sort P[1...N-1].
        # This will always sort the array.
        
        # Check if already sorted
        if all(p[i] == i + 1 for i in range(n)):
            return 0
        
        # Check if 1 operation suffices
        # We need a k such that P[k-1] == k and 
        # max(P[0...k-2]) == k-1 and min(P[k...n-1]) == k+1
        # To do this without loops, we can use prefix maxs and suffix mins.
        # But we can't use loops to build them. 
        # We can use a trick with list comprehensions and a helper.
        
        # Instead of loops, we use a generator expression inside all()
        # But we need prefix/suffix info. Let's use a different logic:
        # An operation k sorts everything except P_k.
        # For the result to be sorted, we must have P_k = k,
        # and the remaining elements must be the correct sets.
        # This means for some k, the set {P_1...P_{k-1}} is {1...k-1}
        # and the set {P_{k+1}...P_N} is {k+1...N}.
        # This is true if and only if P_k = k and 
        # (for all i < k, P_i < k) and (for all i > k, P_i > k).
        
        # We can check this by:
        # 1. Finding all k where P[k-1] == k.
        # 2. For those k, checking the range condition.
        # To avoid loops, we use a list comprehension to find valid k's.
        # To check the range condition without loops, we can use:
        # sum(1 for i in range(k-1) if P[i] >= k) == 0 AND
        # sum(1 for i in range(k, n) if P[i] <= k) == 0
        
        # However, the sum() inside the comprehension is still a loop.
        # The constraint says "no for/while loops". List comprehensions are allowed.
        # The logic:
        # result = 0 if sorted
        # result = 1 if exists k such that P[k-1]==k and all(P[i]<k for i<k-1) and all(P[i]>k for i>=k)
        # result = 2 otherwise.
        
        # We can use a helper to check the "1" condition.
        # Since we can't use loops, we use a list comprehension to check all k.
        
        # To make it efficient, we only check k where P[k-1] == k.
        # We use a generator expression inside any().
        
        # Note: the 'all' and 'any' functions are allowed.
        
        is_sorted = all(p[i] == i + 1 for i in range(n))
        if is_sorted: return 0
        
        # Check for k in 1...N
        # We use a list comprehension to iterate k and 'all' to check conditions.
        can_do_1 = any(
            (p[k-1] == k and 
             all(p[i] < k for i in range(k-1)) and 
             all(p[i] > k for i in range(k, n)))
            for k in range(1, n + 1)
        )
        
        return 1 if can_do_1 else 2

    # Execute the logic
    results = process_cases(input_data)
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()