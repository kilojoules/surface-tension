The constraint to avoid explicit loops forces a shift toward a functional programming paradigm, emphasizing declarative data transformations over imperative state changes. While not always more "idiomatic" for every task, it leverages Python's powerful built-in iterators and higher-order functions, often leading to more concise code for sequence processing. To solve this problem without loops, I will use `functools.reduce` to maintain a state (a stack of indices) as I traverse the buildings from right to left, and list comprehensions/`map` for formatting the output.

```python
import sys
from functools import reduce

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j"
    # means we are looking for the number of elements to the right of i
    # that are "visible" if we look from i.
    # Specifically, j satisfies the condition if H[k] < H[j] for all i < k < j.
    # This is equivalent to saying that the sequence of heights forming the 
    # "running maximums" starting from i+1 determines the count.
    # However, a simpler way to think about it: 
    # For a fixed i, we want to count j > i such that max(H[i+1]...H[j-1]) < H[j].
    # This is exactly the number of elements in a monotonic stack when 
    # processing from right to left.
    
    # We process the array from right to left. 
    # We maintain a stack of indices whose heights are strictly increasing 
    # (when viewed from right to left, so decreasing heights).
    # For a building i, the buildings j > i that satisfy the condition are 
    # exactly those that would remain in a monotonic stack of heights 
    # encountered from i+1 to N.
    
    # Using reduce to simulate the process without explicit for/while loops.
    # State: (stack, results)
    # We iterate through the heights in reverse order.
    def accumulate_visible(state, height):
        stack, results = state
        # Remove elements from the stack that are smaller than the current height
        # because they will be hidden by the current building for any building to the left.
        # Since we can't use 'while', we use a helper function or a clever slice.
        # But wait, the condition is: no building taller than Building j between i and j.
        # This means Building j must be a 'prefix maximum' of the sequence H[i+1...N].
        # For a fixed i, the valid j's are those where H[j] > max(H[i+1...j-1]).
        # This is simply the number of elements in a monotonic stack of heights 
        # processed from index i+1 upwards.
        pass

    # Correct logic:
    # For a fixed i, we are looking for j > i such that H[j] > max(H[i+1...j-1]).
    # This is equivalent to the number of elements in the monotonic stack 
    # (increasing) if we process the array from i+1 to N.
    # However, we need this for ALL i.
    # Let's process from right to left. For index i, the buildings j > i 
    # that satisfy the condition are those that are "visible" from i.
    # A building j is visible from i if H[j] is greater than all H[k] for i < k < j.
    # This means the valid j's are the indices in the monotonic stack 
    # (strictly increasing heights) when scanning from i+1 to N.
    # Actually, the simplest way: for a fixed i, the valid j's are the 
    # elements of the monotonic stack maintained while iterating from i+1 to N.
    # But we need it for all i. 
    # Let's use the property: j is counted for i if H[j] > max(H[i+1...j-1]).
    # This is exactly the number of elements in the monotonic stack 
    # when we process the array from right to left and maintain a stack 
    # of heights that are strictly increasing.
    
    # Let's redefine: for a fixed i, we want to count j > i such that 
    # H[j] is a left-to-right maximum of the suffix H[i+1...N].
    # This is not quite right. Let's re-read: "no building taller than Building j 
    # between Buildings i and j".
    # For i=1, H=[2, 1, 4, 3, 5]. j=2 (H=1): ok. j=3 (H=4): ok (1 < 4). 
    # j=4 (H=3): not ok (4 > 3). j=5 (H=5): ok (max(1,4,3) < 5).
    # The valid j's are those that form a strictly increasing subsequence 
    # of the suffix H[i+1...N] if we only keep elements that are 
    # greater than all previous elements in that suffix.
    
    # This is solved by: for each i, the answer is the size of the 
    # monotonic stack of the suffix H[i+1...N] processed from left to right.
    # Wait, that's still a loop. Let's use the property:
    # j is counted for i if H[j] > max(H[i+1...j-1]).
    # This means for a fixed j, it is counted for all i < j such that 
    # max(H[i+1...j-1]) < H[j].
    # This is equivalent to saying i must be greater than the index of the 
    # first building to the left of j that is taller than H[j].
    # Let L[j] be the index of the nearest building to the left of j such that H[L[j]] > H[j].
    # If no such building exists, L[j] = 0.
    # Then j is counted for all i such that L[j] <= i < j.
    # The number of such i's is j - L[j].
    # But we need the count for each i.
    # For a fixed i, we need the number of j > i such that L[j] <= i.
    
    # Let's use the property: the answer for i is the number of j > i 
    # such that H[j] > max(H[i+1...j-1]).
    # This is exactly the size of the monotonic stack when processing 
    # the suffix H[i+1...N].
    # Another way: the answer for i is the number of elements in the 
    # monotonic stack (increasing) of the suffix H[i+1...N].
    # This can be solved by processing from right to left.
    # When moving from i+1 to i, we add H[i+1] to the front of the 
    # sequence. The new monotonic stack will be H[i+1] followed by 
    # the elements of the previous monotonic stack that are taller than H[i+1].
    
    def step(state, h):
        stack, results = state
        # New stack: current height h, followed by elements in the 
        # existing stack that are taller than h.
        # Since the stack is already sorted (bottom is tallest, top is shortest 
        # when viewed from the right), we can use binary search.
        import bisect
        # The stack contains heights of buildings to the right that were 
        # visible. It will be sorted in increasing order.
        # We need to find how many elements in the current stack are > h.
        idx = bisect.bisect_right(stack, h)
        count = len(stack) - idx
        # The new stack for the next (leftward) building will be 
        # h combined with the elements of the current stack that are > h.
        # However, the "stack" for the next building (i-1) is the 
        # monotonic stack of the suffix starting at i.
        # The elements of the monotonic stack for suffix i are:
        # H[i] and all elements of the monotonic stack of suffix i+1 that are > H[i].
        return ( [h] + stack[idx:], [count] + results )

    # We process from right to left.
    # For the last building (N), the answer is always 0.
    # For building N-1, the answer is 1 (building N).
    # We use reduce to iterate backwards through the heights.
    final_stack, final_results = reduce(
        lambda state, h: step(state, h),
        reversed(H),
        ([], [])
    )
    
    # The reduce processes H_N, H_{N-1}... H_1.
    # For H_N: stack=[], res=[0]. Next stack=[H_N].
    # For H_{N-1}: stack=[H_N], res=[count, 0]. Next stack=[H_{N-1}, ...].
    # The results list will be in order [c_1, c_2, ... c_N].
    # Because we processed right-to-left and prepended to the results list.
    # Wait, the 'step' function logic:
    # For the very last element H[N-1], the stack is empty, count is 0.
    # Then stack becomes [H[N-1]].
    # For H[N-2], we check how many in [H[N-1]] are > H[N-2].
    # That is the answer for i = N-2.
    
    # Let's refine the reduce:
    # The first element processed is H[N-1]. Its answer is 0.
    # The second is H[N-2]. Its answer is the number of elements in the 
    # monotonic stack of H[N-1...N-1] that are > H[N-2], plus 1 (for H[N-1] itself).
    # Actually, the condition is: j > i and no building between i and j is taller than H[j].
    # This means H[j] > max(H[i+1...j-1]).
    # For a fixed i, the j's are exactly the indices that would form the 
    # monotonic increasing stack of the suffix H[i+1...N].
    # The size of this stack is the answer.
    
    # Correct step:
    # 1. Current building is H[i].
    # 2. The buildings j > i that satisfy the condition are:
    #    Building i+1, and any building j > i+1 that is taller than 
    #    all buildings between i+1 and j, AND taller than H[i+1].
    #    This is exactly: 1 (for building i+1) + (number of elements in the 
    #    monotonic stack of suffix i+1 that are taller than H[i+1]).
    # Wait, the condition is about buildings BETWEEN i and j.
    # For j = i+1, there are no buildings between, so it always satisfies.
    # For j > i+1, it satisfies if H[j] > max(H[i+1...j-1]).
    # This means j must be part of the monotonic increasing stack of the suffix H[i+1...N].
    # The number of such j's is simply the size of the monotonic stack of the suffix H[i+1...N].
    
    # Let's trace Sample 1: 2 1 4 3 5
    # i=5: ans=0. Stack=[]
    # i=4: H[4]=3. Suffix [5]. Monotonic stack of suffix: [5]. Size=1. Ans=1.
    # i=3: H[2]=4. Suffix [3, 5]. Monotonic stack: [3, 5]. Size=2. Ans=2.
    # i=2: H[1]=1. Suffix [4, 3, 5]. Monotonic stack: [4, 5]. Size=2. Ans=2.
    # i=1: H[0]=2. Suffix [1, 4, 3, 5]. Monotonic stack: [1, 4, 5]. Size=3. Ans=3.
    # Results: 3 2 2 1 0. Correct.
    
    # To implement this:
    # For i = N down to 1:
    #   ans[i] = len(monotonic_stack(H[i+1...N]))
    #   monotonic_stack(H[i...N]) = [H[i]] + {x in monotonic_stack(H[i+1...N]) if x > H[i]}
    
    def step_final(state, h):
        stack, results = state
        # The answer for