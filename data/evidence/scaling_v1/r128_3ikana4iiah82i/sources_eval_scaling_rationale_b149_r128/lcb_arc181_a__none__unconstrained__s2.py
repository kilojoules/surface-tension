The constraint to avoid explicit loops forces a shift toward functional programming paradigms, emphasizing the transformation of data streams over stateful iteration. In idiomatic Python, this encourages the use of high-level abstractions like list comprehensions and `map`, which are often more concise and can be more efficient. For this problem, I will use list comprehensions to process the test cases and `functools.reduce` or `map` to handle the logic of checking if the permutation is already sorted or can be solved in one operation.

```python
import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    
    # Use a generator to yield the start index of each test case
    # Since N varies, we need to track the current pointer in the input_data list
    # However, since we cannot use loops, we use a recursive-like structure 
    # via a custom function and map, or we pre-process the input into chunks.
    
    # To avoid loops, we first determine the boundaries of each test case.
    # We use a helper function to group the input into (N, P) pairs.
    def group_cases(data):
        # We use a list comprehension to build the cases.
        # Because N is variable, we can't use a simple slice.
        # But we can use a trick with a mutable list and a side-effect in a comprehension.
        # Actually, the most reliable way without loops is to use a 
        # recursive-like approach with a function that slices the list.
        pass

    # Since we cannot use 'for' or 'while', we use map and a helper 
    # to process the flat list.
    
    # Let's redefine: we need to process T cases.
    # We can use a recursive-style approach using a function and 
    # slicing, but recursion depth is an issue.
    # The best way to handle variable length chunks without loops 
    # is to use a generator and next().
    
    it = iter(input_data)
    
    def process_case():
        try:
            N = int(next(it))
            P = [int(next(it)) for _ in range(N)]
            
            # The goal is to find the minimum operations to make P_i = i.
            # 0 ops: Already sorted.
            # 1 op: There exists k such that sorting [1, k-1] and [k+1, N] 
            #       results in [1, ..., N].
            #       This happens if there is exactly one element P_k that is 
            #       "out of place" in a way that the others can be sorted around it.
            #       Actually, 1 op is possible if there exists k such that:
            #       {P_1, ..., P_{k-1}} = {1, ..., k-1} AND
            #       {P_{k+1}, ..., P_N} = {k+1, ..., N}.
            #       This implies P_k must be k.
            #       Wait, the operation sorts the ranges. If we pick k, 
            #       the result is sorted if the set of elements in the 
            #       first range is {1, ..., k-1} and the second is {k+1, ..., N}.
            #       This is true if and only if P_k = k.
            #       BUT, the operation sorts the ranges. So if we pick k, 
            #       the final array is sorted if and only if 
            #       the elements {P_1, ..., P_{k-1}} are some permutation of {1, ..., k-1}
            #       and {P_{k+1}, ..., P_N} are some permutation of {k+1, ..., N}.
            #       This is equivalent to saying P_k = k.
            #       Wait, if P_k = k, then the elements in the first range 
            #       must be {1, ..., k-1} and the second {k+1, ..., N} 
            #       for the whole thing to be sorted.
            #       Actually, the condition for 1 op is:
            #       There exists k such that P_k = k AND 
            #       max(P_1, ..., P_{k-1}) < k AND min(P_{k+1}, ..., P_N) > k.
            #       This is simply P_k = k. 
            #       Wait, if P_k = k, then the elements in the first range 
            #       are not necessarily {1, ..., k-1}. 
            #       Example: P = [2, 1, 3, 5, 4], k=3. 
            #       P_3 = 3. First range [2, 1], second [5, 4].
            #       Sorting them gives [1, 2, 3, 4, 5]. Correct.
            #       So 1 op is possible if there exists k such that P_k = k.
            #       Is that always true? No. If P = [2, 3, 1], no P_k = k.
            #       If P = [1, 2, 3], 0 ops.
            #       If P = [2, 1, 3, 5, 4], k=3 works.
            #       What if P = [3, 2, 1, 7, 5, 6, 4]? 
            #       P_1=3, P_2=2, P_3=1, P_4=7, P_5=5, P_6=6, P_7=4.
            #       P_2=2, P_5=5, P_6=6 are the only candidates for k.
            #       k=2: [3, 1] sorted is [1, 3], [7, 5, 6, 4] sorted is [4, 5, 6, 7].
            #       Result: [1, 3, 2, 4, 5, 6, 7]. Not sorted.
            #       k=5: [3, 2, 1, 7] sorted is [1, 2, 3, 7], [6, 4] sorted is [4, 6].
            #       Result: [1, 2, 3, 7, 5, 4, 6]. Not sorted.
            #       k=6: [3, 2, 1, 7, 5] sorted is [1, 2, 3, 5, 7], [4] sorted is [4].
            #       Result: [1, 2, 3, 5, 7, 6, 4]. Not sorted.
            #       So for [3, 2, 1, 7, 5, 6, 4], answer is 2.
            
            # Correct condition for 1 op:
            # There exists k such that:
            # (Sorted P[0:k-1]) + [P[k-1]] + (Sorted P[k:N]) == [1, ..., N]
            # This means:
            # 1. P[k-1] must be k.
            # 2. The set {P_0, ..., P_{k-2}} must be {1, ..., k-1}.
            # 3. The set {P_k, ..., P_{N-1}} must be {k+1, ..., N}.
            # Actually, if P[k-1] = k, then condition 2 and 3 are automatically 
            # satisfied because P is a permutation of 1...N.
            # If P[k-1] = k, then the remaining N-1 elements are {1...N} \ {k}.
            # The first k-1 elements must be {1...k-1} for the result to be sorted.
            # So the condition for 1 op is:
            # There exists k such that P[k-1] = k AND max(P[0...k-2]) < k.
            # (Note: if k=1, max is -inf; if k=N, max is P[0...N-2]).
            
            # Let's refine:
            # 0 ops: P == sorted(P)
            # 1 op: There exists k in 1...N such that:
            #       (k == 1 or max(P[0:k-1]) < k) AND 
            #       (k == N or min(P[k:N]) > k)
            #       Actually, if P[k-1] == k, then the condition 
            #       "max(P[0:k-1]) < k" is equivalent to saying 
            #       all elements before index k-1 are less than k.
            
            # To implement this without loops:
            # Use list comprehensions to find all k that satisfy the condition.
            
            # We need prefix max and suffix min.
            # Since we can't use loops, we can't use a standard prefix max array.
            # But we can use a trick with a list and a function.
            # Wait, the constraint says no 'for' or 'while'. 
            # It does NOT forbid list comprehensions.
            # But prefix max requires the previous value.
            # We can use a recursive-like structure with a helper 
            # or just use the fact that N is 2*10^5 and we can't use loops.
            # Actually, we can use a technique to simulate prefix max 
            # using a list and a side-effect in a list comprehension, 
            # but that's hacky.
            # The cleanest way is to use a function that processes the list.
            # But I can't use loops. 
            # I will use a method to calculate prefix max and suffix min 
            # using a Divide and Conquer approach or 
            # by using a library if allowed (but only standard libs).
            # Actually, I can use a list comprehension with a 
            # helper function that uses slicing, but that's O(N^2).
            # Wait! I can use a list comprehension to build the prefix max 
            # if I use a mutable object to store the current max.
            
            # Let's use a class to hold the state.
            class State:
                def __init__(self, vals):
                    self.vals = vals
                    self.cur = -1
                def get_prefix_max(self, x):
                    self.cur = max(self.cur, x)
                    return self.cur
            
            class StateMin:
                def __init__(self, vals):
                    self.vals = vals
                    self.cur = float('inf')
                def get_suffix_min(self, x):
                    self.cur = min(self.cur, x)
                    return self.cur
            
            p_max = [State(P).get_prefix_max(x) for x in P] # This is wrong, State is recreated
            # Correct way:
            s_max = State(P)
            p_max = [s_max.get_prefix_max(x) for x in P]
            
            s_min = StateMin(P[::-1])
            p_min = [s_min.get_suffix_min(x) for x in P[::-1]][::-1]
            
            # Now check conditions:
            # 0 ops:
            is_sorted = (P == sorted(P))
            
            # 1 op:
            # For k from 1 to N:
            # Condition: (k==1 or p_max[k-2] < k) and (k==N or p_min[k] > k) and P[k-1] == k
            # Note: p_max[k-2] is the max of first k-1 elements.
            # p_min[k] is the min of elements from index k to N-1.
            
            # We can use a list comprehension to check if any k satisfies this.
            # We need to handle boundaries k=1 and k=N.
            
            # For k=1: P[0] == 1 and (N==1 or p_min[1] > 1)
            # For k=N: P[N-1] == N and (N==1 or p_max[N-2] < N)
            # For 1 < k < N: P[k-1] == k and p_max[k-2] < k and p_min[k] > k
            
            # We can generalize this by padding p_max and p_min.
            # p_max: [-1, ...], p_min: [..., inf]
            
            # Let's use a simpler check for 1 op:
            # A k works if P[k-1] == k and the set of elements 
            # {P_0, ..., P_{k-2}} is {1, ..., k-1}.
            # This is true if p_max[k-2