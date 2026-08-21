```python
import sys
from itertools import groupby

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    
    # Use a generator to process test cases without explicit for-loops
    # We use a helper function to extract the N and P for each case
    def get_cases(data):
        it = iter(data[1:])
        return ( (int(next(it)), [int(next(it)) for _ in range(int(current_n))]) 
                 for current_n in (lambda it_inner: (next(it_inner) for _ in range(T)))(it) )
    
    # This is a bit tricky because the generator expression above consumes the iterator.
    # Let's redefine the case extraction to be more robust.
    def case_generator(data):
        it = iter(data[1:])
        def process():
            try:
                n_str = next(it)
                n = int(n_str)
                p = [int(next(it)) for _ in range(n)]
                yield (n, p)
            except StopIteration:
                pass
        
        # To avoid for-loops, we use a recursive-like structure via map/reduce or 
        # a list comprehension that drives the iterator.
        # Since we need to do this T times, we can use a list comprehension.
        return [next(case_gen) for case_gen in ( ( (yield from [ (n, [int(next(it)) for _ in range(n)]) ]) 
                  for n in (lambda it_local: [int(next(it_local)) for _ in range(T)])(it) ) )]

    # Actually, the simplest way to avoid 'for' is to use a list comprehension 
    # and a helper that manages the pointer.
    # But since we can't use 'for', we use a closure and a list to track the index.
    
    # Let's use a different approach: flatten the input and use slicing.
    # But N varies, so we can't slice uniformly.
    # We can use a recursive-like structure with map and a mutable state.
    
    def run():
        # Use a list to store the current index to simulate a pointer
        # We use a list comprehension to iterate T times.
        # We use a helper function to extract the permutation.
        
        # Since we cannot use 'for', we use 'map' and 'list comprehensions'.
        # To handle the variable N, we can use a trick with a mutable object.
        state = {'ptr': 1}
        
        def extract_case():
            n = int(input_data[state['ptr']])
            state['ptr'] += 1
            p = input_data[state['ptr'] : state['ptr'] + n]
            state['ptr'] += n
            return (n, p)

        # Process T cases
        results = list(map(lambda _: extract_case(), range(T)))
        
        # For each (N, P), calculate the answer:
        # The answer is 0 if already sorted.
        # The answer is 1 if there exists k such that sorting [1, k-1] and [k+1, N] sorts the whole array.
        # This happens if there is exactly one element P_k that is 'out of place' in a way that
        # all elements to its left are <= P_k and all to its right are >= P_k, 
        # AND the remaining elements are already partitioned correctly.
        # Actually, the condition for 1 operation is:
        # There exists k such that {P_1...P_{k-1}} = {1...k-1} and {P_{k+1}...P_N} = {k+1...N}.
        # This is equivalent to saying P_i = i for all i != k.
        
        # Wait, the operation sorts the ranges. So if we pick k, 
        # the result is sorted if and only if the set of values {P_1...P_{k-1}} is {1...k-1}
        # and the set of values {P_{k+1}...P_N} is {k+1...N}.
        # This means P_i = i for all i except possibly i = k.
        
        # Let's refine:
        # 0 ops: P_i = i for all i.
        # 1 op: There exists k such that for all i != k, P_i = i.
        # 2 ops: Otherwise. (It's proven that 2 is always enough for N >= 3).
        
        def evaluate(case):
            n, p = case
            # Count how many i have P_i != i
            # Use a list comprehension to find indices where P_i != i
            diffs = [i for i, val in enumerate(p, 1) if val != str(i)]
            diff_count = len(diffs)
            
            if diff_count == 0:
                return 0
            if diff_count <= 1:
                return 1
            
            # Special case for 1 op: if diff_count == 2, say at indices i and j,
            # we can only solve it in 1 op if one of them is the 'k' and the other is already in place.
            # But if P_i != i and P_j != j, then for any k, at least one of i or j will be in a 
            # sorted range. If i is in [1, k-1], it will be sorted to its correct position.
            # So if we pick k=j, then P_i will be sorted to i, and P_j stays at j.
            # If P_j was already j, diff_count would be 1.
            # If P_j != j, then after sorting [1, j-1] and [j+1, N], 
            # the only element that could be misplaced is P_j.
            # For the whole thing to be sorted, we need P_j to be j.
            # But we assumed P_j != j.
            # Therefore, if diff_count >= 2, it takes 2 operations.
            # UNLESS: the operation allows us to fix everything.
            # Let's re-read: "sort the 1-st through (k-1)-th terms... sort the (k+1)-th through N-th".
            # If we pick k, then P_k remains unchanged. All other P_i are sorted.
            # The result is P_i = i for all i iff:
            # 1. P_k = k
            # 2. The set {P_1...P_{k-1}} is {1...k-1}
            # 3. The set {P_{k+1}...P_N} is {k+1...N}
            # This is exactly the condition: P_i = i for all i != k.
            # If P_k = k and P_i = i for all i != k, then diff_count = 0.
            # If P_k != k and P_i = i for all i != k, then diff_count = 1.
            # In both cases, 1 operation (with that k) suffices (or 0).
            # If diff_count >= 2, then for any k, at least one i != k will have P_i != i.
            # Wait, that's not right. If P = [2, 1, 3], and we pick k=3, 
            # we sort [1, 2], getting [1, 2, 3]. Here diff_count was 2 (i=1, 2).
            # So if diff_count >= 2, we can still succeed in 1 op if there exists k such that
            # P_k = k and the remaining elements are just a permutation of the remaining values.
            # Actually, the condition for 1 op is:
            # There exists k such that P_k = k, and sorting the others fixes them.
            # Sorting the others always fixes them if the sets are correct.
            # The sets {P_1...P_{k-1}} and {P_{k+1}...P_N} are correct iff P_k = k.
            # Because the total set is {1...N}, if P_k = k, the remaining must be {1...N} \ {k}.
            # So 1 op is possible iff there exists k such that P_k = k.
            # If P_i != i for all i, then 2 ops.
            # If P_i = i for all i, then 0 ops.
            # If some P_i = i, then 1 op.
            
            # Let's double check Sample 1: [2, 1, 3, 5, 4]. P_3 = 3. Answer: 1. Correct.
            # Sample 2: [1, 2, 3]. All P_i = i. Answer: 0. Correct.
            # Sample 3: [3, 2, 1, 7, 5, 6, 4]. P_2=2, P_6=6. Wait, Sample 3 says 2.
            # Let me re-read. "sort the 1-st through (k-1)-th... sort the (k+1)-th through N-th".
            # If P = [3, 2, 1, 7, 5, 6, 4] and k=2, P becomes [3, 2, 1, 4, 5, 6, 7].
            # Then P_2 is still 2, but P_1 is 3. Not sorted.
            # My logic "P_k = k" is necessary but not sufficient.
            # The condition is: sorting [1, k-1] and [k+1, N] must result in P_i = i.
            # This happens iff:
            # 1. P_k = k
            # 2. {P_1, ..., P_{k-1}} = {1, ..., k-1}
            # 3. {P_{k+1}, ..., P_N} = {k+1, ..., N}
            # This is equivalent to: P_k = k AND (for all i < k, P_i < k) AND (for all i > k, P_i > k).
            
            # Let's check Sample 3 again: [3, 2, 1, 7, 5, 6, 4]
            # k=1: P_1=3 (no)
            # k=2: P_2=2. Left: {3}, Right: {1, 7, 5, 6, 4}. Left is not {1}. (no)
            # k=3: P_3=1 (no)
            # k=4: P_4=7 (no)
            # k=5: P_5=5. Left: {3, 2, 1, 7}, Right: {6, 4}. Left is not {1, 2, 3, 4}. (no)
            # k=6: P_6=6. Left: {3, 2, 1, 7, 5}, Right: {4}. Right is not {7}. (no)
            # k=7: P_7=4 (no)
            # All k fail. Answer 2. Correct.
            
            # So the condition for 1 op is:
            # Exists k such that P_k = k and max(P_1...P_{k-1}) < k and min(P_{k+1}...P_N) > k.
            # This is equivalent to: P_k = k and max(P_1...P_{k-1}) = k-1 and min(P_{k+1}...P_N) = k+1.
            # (With boundary conditions for k=1 and k=N).
            
            # To implement this without loops:
            # 1. Create a list of P_i = i.
            # 2. Use a list comprehension to find all k satisfying the condition.
            # 3. We need prefix max and suffix min.
            # Since we can't use loops, we can't use reduce for prefix/suffix.
            # But we can use a trick: the condition "max(P_1...P_{k-1}) < k and min(P_{k+1}...P_N) > k"
            # is equivalent to saying that the set {P_1...P_{k-1}} is exactly {1...k-1}.
            # This is true iff max(P_1...P_{k-1}) == k-1.
            
            # Wait, we can't use loops, but we can use a list comprehension to 
            # check if the permutation is already sorted.
            # For the 1-op case, we need to check if any k works.
            #