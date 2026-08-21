The constraint to avoid explicit loops forces a shift toward functional programming paradigms, emphasizing declarative data transformations over imperative state changes. While not always more "idiomatic" for every task, it leverages Python's powerful built-in iterators and higher-order functions, often leading to more concise code for sequence processing. To solve this problem without loops, I will use a recursive approach combined with a stack (implemented via a helper function) to simulate the monotonic stack algorithm required to count visible buildings, as Python's recursion limit must be handled for $N=2 \times 10^5$. However, since true recursion is risky for this scale, I will use `functools.reduce` to maintain a state (the stack and the result list) across the sequence, effectively simulating a loop using a functional fold.

```python
import sys
from functools import reduce

def solve():
    # Increase recursion depth just in case, though reduce is used
    sys.setrecursionlimit(300000)
    
    # Read N and heights
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # The condition "no building taller than Building j between i and j"
    # means we are looking for the number of elements to the right of i
    # that are "visible". An element j is visible from i if 
    # max(H[i+1]...H[j-1]) < H[j].
    # This is equivalent to saying that if we process the array from right to left,
    # for a fixed i, we want to count how many j > i are such that 
    # H[j] is a left-to-right maximum of the suffix starting at i+1.
    # Wait, the condition is: for a fixed i, j satisfies it if 
    # for all k such that i < k < j, H[k] < H[j].
    # This means Building j must be taller than all buildings between i and j.
    # This is exactly the definition of elements that would remain in a 
    # monotonic stack if we processed the array from j down to i.
    # Actually, for a fixed i, the indices j that satisfy this are the 
    # indices of the upper-convex hull/monotonic increasing sequence 
    # starting from the first element to the right of i.
    # Specifically, j1 = i+1, and j_{m+1} is the first index > j_m such that H[j_{m+1}] > H[j_m].
    # No, that's not correct. The condition is: H[k] < H[j] for all i < k < j.
    # This means Building j is visible from i if it is taller than all buildings between them.
    # This is equivalent to saying that in the range [i+1, j], H[j] is the maximum.
    
    # Correct logic: For a fixed i, we are looking for j > i such that 
    # H[j] > max(H[i+1], ..., H[j-1]).
    # This means we are counting the number of "prefix maximums" of the suffix H[i+1:].
    # Let's process from right to left. When we are at index i, we want to know
    # how many j > i are prefix maximums of H[i+1:].
    # If we maintain a monotonic stack of heights for the suffix, the number of
    # visible buildings for i is simply the size of the monotonic stack 
    # constructed from the suffix H[i+1:], but only those elements that 
    # are taller than any element encountered between i and them.
    # Actually, the simplest way: j satisfies the condition if H[j] > max(H[i+1...j-1]).
    # This means for a fixed i, the valid j's are:
    # j_1 = i+1
    # j_2 = first index > j_1 such that H[j_2] > H[j_1]
    # j_3 = first index > j_2 such that H[j_3] > H[j_2]...
    # This is exactly the number of elements in a monotonic increasing stack 
    # when processing the suffix H[i+1:] from left to right.
    # But we need this for all i.
    # Let's use the property: j is visible from i if H[j] is a 
    # "right-side" maximum. 
    # For a fixed j, it is visible from all i < j such that 
    # max(H[i+1...j-1]) < H[j].
    # This means i must be such that all buildings between i and j are shorter than H[j].
    # Let L[j] be the index of the first building to the left of j that is taller than H[j].
    # Then j is visible from all i in the range [L[j], j-1].
    # The number of such i is j - L[j].
    # We want for each i, the count of j > i such that i >= L[j].
    # This is a range update problem: for each j, add 1 to range [L[j], j-1].
    # Then query values at each i.
    
    # To avoid loops, we use reduce to find L[j] and another reduce to aggregate counts.
    # L[j] can be found using a monotonic stack.
    
    def find_L(acc, x):
        stack, L = acc
        # Pop elements smaller than current height
        # We use a helper function to simulate the while loop for popping
        def pop_smaller(s):
            return s[:-1] if s and s[-1][0] < x else s
        
        # Since we can't use while, we use a recursive-like approach via a 
        # custom function or just a trick with a list.
        # However, the constraint says no loops. We can use a recursive function 
        # to handle the popping logic.
        return (stack, L)

    # Since we cannot use loops or recursion (for large N), 
    # and we must use functional style, we use a trick:
    # We can use a list and a pointer, but that's basically a loop.
    # The only way to truly avoid loops/recursion for this logic is 
    # to use a data structure or a very clever reduce.
    # Actually, the most reliable way to implement a monotonic stack 
    # without 'for/while' is to use a recursive function with 
    # @lru_cache or similar, but N is too large.
    # Wait, the constraint says "no for or while loops". 
    # It does NOT forbid recursion if implemented carefully, 
    # but Python's recursion limit is an issue.
    # Let's use a different approach: 
    # The number of j's for a given i is the number of elements in the 
    # monotonic stack of the suffix H[i+1:].
    # If we process from right to left, we can maintain the monotonic stack.
    # When moving from i+1 to i, we are adding H[i+1] to the front of the suffix.
    # The new monotonic stack will be H[i+1] followed by all elements in the 
    # previous stack that are taller than H[i+1].
    
    # Let S_{i+1} be the monotonic stack of the suffix H[i+1:].
    # S_i = [H[i+1]] + [x for x in S_{i+1} if x > H[i+1]]
    # The answer for i is len(S_i).
    
    # We can implement this using reduce, processing the array in reverse.
    # The state will be (current_stack, results_list).
    
    def step(state, h_i_plus_1):
        stack, results = state
        # New stack is h_i_plus_1 followed by elements in stack > h_i_plus_1
        # To avoid a loop/list comprehension (which is a loop), 
        # we can use filter().
        new_stack = [h_i_plus_1] + list(filter(lambda x: x > h_i_plus_1, stack))
        return (new_stack, [len(new_stack)] + results)

    # We need to process H from N-1 down to 1.
    # For i = N, the answer is 0.
    # For i = N-1, the suffix is H[N-1:], the stack is [H[N-1]], count is 1.
    # Wait, the index i is 1-based. For i=N, c_N = 0.
    # For i=N-1, j=N is the only possibility. H[N] is always > max(empty), so c_{N-1} = 1.
    
    # Let's refine:
    # We process the heights in reverse order: H[N-1], H[N-2]... H[0]
    # For the last element H[N-1], it's the suffix for i=N-1.
    # But the problem asks for i=1 to N.
    # For i=N: c_N = 0.
    # For i=N-1: suffix is H[N-1:], stack is [H[N-1]], count is 1.
    # For i=N-2: suffix is H[N-1:], but we need to consider j > i.
    # The buildings to the right of i are H[i+1...N-1].
    # Let's use the property: c_i = length of the monotonic increasing stack of H[i+1...N-1].
    
    # Correct sequence:
    # i = N: suffix is empty, c_N = 0
    # i = N-1: suffix is H[N-1], stack is [H[N-1]], c_{N-1} = 1
    # i = N-2: suffix is H[N-1...N-1], we need the monotonic stack of that.
    # Actually, let's process from right to left.
    # Let f(i) be the monotonic stack of the suffix H[i...N-1].
    # f(N) = []
    # f(i) = [H[i]] + [x for x in f(i+1) if x > H[i]]
    # c_i = len(f(i+1))
    
    # Example 1: 2 1 4 3 5
    # i=5: c_5 = 0, f(5) = []
    # i=4: H[4]=5, f(4) = [5] + [], c_4 = len(f(5)) = 0. Wait, the sample says c_4=1.
    # Let's re-read: j > i. For i=4, j=5. H[5]=5. Between 4 and 5 there are no buildings.
    # So j=5 satisfies it. c_4 = 1.
    # My f(i) definition: f(i) is the monotonic stack of H[i...N-1].
    # For i=4, suffix is H[4...4] (0-indexed), which is [5]. Stack is [5], len is 1.
    # So c_i = len(f(i+1)).
    
    # Let's trace Sample 1: 2 1 4 3 5
    # H = [2, 1, 4, 3, 5]
    # i=5 (idx 4): c_5 = 0, f(5) = []
    # i=4 (idx 3): suffix H[4:5] = [5], f(4) = [5], c_4 = len(f(4)) = 1
    # i=3 (idx 2): suffix H[3:5] = [3, 5], f(3) = [3, 5], c_3 = len(f(3)) = 2
    # i=2 (idx 1): suffix H[2:5] = [4, 3, 5], f(2) = [4, 5], c_2 = len(f(2)) = 2
    # i=1 (idx 0): suffix H[1:5] = [1, 4, 3, 5], f(1) = [1, 4, 5], c_1 = len(f(1)) = 3
    # Results: 3 2 2 1 0. Matches sample!
    
    # Implementation using reduce:
    # We process H in reverse.
    # The state is (current_stack, results_list).
    # For the very last element H[N-1], it starts the process.
    
    # To avoid the loop in the slice/filter, we use the fact that 
    # we are processing the list