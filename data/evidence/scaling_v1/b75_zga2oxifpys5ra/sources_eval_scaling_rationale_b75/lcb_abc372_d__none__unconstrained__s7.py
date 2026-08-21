The constraint to avoid explicit loops and recursion forces a shift from imperative programming to a functional paradigm. While idiomatic Python often uses `for` loops for readability, utilizing `itertools` and `functools.reduce` allows for concise data transformations and state accumulation. For this specific problem, we need to maintain a "monotonic stack" to count visible buildings, which is naturally handled by `reduce` as it passes an accumulator (the stack and the result list) through the sequence.

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
    # means we are looking for the number of indices j > i such that
    # H[k] < H[j] for all i < k < j.
    # This is equivalent to saying that if we process the array from right to left,
    # for a fixed i, we want to count how many j > i are "visible".
    # A building j is visible from i if it is taller than all buildings between i and j.
    # This is exactly the number of elements in a monotonic increasing stack 
    # when iterating backwards from N-1 down to i.
    
    # We use reduce to maintain a state: (stack, results)
    # stack: stores heights of buildings that could be the 'j' for future 'i's
    # results: stores the count of visible buildings for each i
    
    # When moving from i+1 to i:
    # The buildings j > i that satisfy the condition are:
    # 1. Building i+1
    # 2. Any building j > i+1 that was visible from i+1 AND is taller than H[i+1]
    # Actually, the simpler logic: j satisfies the condition if H[j] > max(H[i+1...j-1]).
    # This means the set of such j's are the elements of the monotonic increasing stack
    # constructed by iterating from i+1 to N.
    
    # Correct logic: For a fixed i, we are looking for j > i such that 
    # H[j] > max(H[i+1], ..., H[j-1]).
    # This is simply the number of elements in a monotonic increasing stack 
    # processed from index i+1 to N-1.
    # However, the problem asks for this for ALL i.
    # Let's reconsider: j is counted for i if H[j] is a prefix maximum of the 
    # sequence H[i+1], H[i+2], ..., H[N].
    
    # Since we need to do this for all i, and N is 2*10^5, an O(N^2) approach is too slow.
    # We need a more efficient way.
    # Wait, the condition is: no building taller than Building j between i and j.
    # This means for a fixed j, it is counted for all i < j such that 
    # H[k] < H[j] for all i < k < j.
    # This means i must be greater than the index of the first building to the left of j 
    # that is taller than H[j].
    # Let L[j] be the index of the nearest building to the left of j such that H[L[j]] > H[j].
    # If no such building exists, L[j] = 0 (using 1-based indexing).
    # Then for a fixed j, the indices i that satisfy the condition are i = L[j], L[j]+1, ..., j-1.
    # Wait, the condition is "between i and j". 
    # If i = L[j], the buildings between are i+1, ..., j-1. All are < H[j]. So i=L[j] works.
    # If i < L[j], then building L[j] is between i and j, and H[L[j]] > H[j], so it fails.
    # So for a fixed j, the valid i's are i \in {L[j], L[j]+1, ..., j-1} AND i >= 1.
    # This means for each j, we add 1 to the count c_i for i in range(max(1, L[j]), j).
    
    # To implement this without loops, we can use a difference array ( Fenwick tree/Segment tree 
    # is overkill, a simple array suffices).
    # 1. Find L[j] for all j using a monotonic stack.
    # 2. Use a difference array to mark ranges [L[j], j-1].
    # 3. Compute prefix sums of the difference array.

    # Step 1: Find L[j] using reduce to simulate a stack
    # State: (stack, L_values) where stack stores indices
    def find_L(state, current_idx):
        stack, L = state
        # Pop from stack while H[top] < H[current_idx]
        # Since we can't use while loops, we use a helper function or a trick.
        # But wait, the constraint says no loops. We can use a recursive-like 
        # structure inside reduce by passing the stack.
        # Actually, the most reliable way to simulate a while loop in reduce 
        # is to use a helper function that handles the popping.
        
        # However, the constraint to avoid loops makes stack popping hard.
        # Let's use the property that we can use list comprehensions and map.
        # But we need the stack state.
        pass

    # Given the strict "no loops" constraint, we must rely on high-order functions.
    # To simulate the monotonic stack's 'while' loop, we can use a recursive-like 
    # approach via a helper function, but recursion is forbidden.
    # The only way to "loop" is via reduce, map, filter, or comprehensions.
    # We can simulate the 'while' loop by using a function that returns 
    # a new stack and the result, and calling it repeatedly? No, that's recursion.
    
    # Let's use the fact that we can use a list comprehension to find the 
    # nearest greater element if we are clever, but that's usually O(N^2).
    # Actually, we can use the 'bisect' module on a sorted list of indices 
    # if we process heights in a certain order, but that's complex.
    
    # Wait, the most idiomatic "no loop" way to handle a monotonic stack 
    # is to use a custom object or a closure within reduce that 
    # manages the stack, but the popping logic still requires a loop.
    # UNLESS we use the fact that we can use `while` inside a function 
    # and just call that function once inside `reduce`. 
    # But the prompt says "no for/while loops". 
    # Let's use a trick: a function that uses `itertools.accumulate` or `reduce`.
    # To simulate a while loop in `reduce`, we can't. 
    # But we can use `sorted` and a Fenwick tree (implemented via a list and 
    # map/reduce) to find L[j].
    
    # Actually, the simplest way to avoid loops and recursion is to use 
    # `numpy` (not allowed) or very clever `itertools`.
    # Let's use the property: L[j] is the index of the first element to the left 
    # larger than H[j]. We can find this by processing elements in 
    # decreasing order of height and using a SortedList (from bisect).
    
    import bisect
    
    # Sort indices by height descending
    sorted_indices = sorted(range(N), key=lambda x: H[x], reverse=True)
    
    # We need to find the nearest index to the left that has already been processed.
    # Since we process in descending order of height, any index already in the 
    # 'processed' set is taller than the current H[j].
    # We maintain a sorted list of indices already processed.
    
    # To avoid loops, we use reduce to maintain the sorted list of processed indices 
    # and a list of L values.
    def process_indices(state, j):
        processed, L = state
        # Find the largest index in 'processed' that is less than j
        idx = bisect.bisect_left(processed, j)
        # The nearest taller building to the left is at processed[idx-1] if idx > 0
        L_j = processed[idx-1] + 1 if idx > 0 else 1
        bisect.insort(processed, j + 1) # Use 1-based indexing for L
        # We need to store L_j for index j. We can't mutate L, so we return a new list?
        # No, that's O(N^2). We can use a dictionary or a mutable list.
        L[j] = L_j
        return (processed, L)

    # Since we are allowed to mutate a list passed in the accumulator:
    L_vals = [0] * N
    reduce(process_indices, sorted_indices,一个 ([], L_vals))
    
    # Now we have L_vals where L_vals[j] is the left bound.
    # The range of i is [L_vals[j], j] (using 1-based indexing).
    # We need to count for each i how many j's satisfy this.
    # This is a range update problem: for each j, add 1 to range [L_vals[j], j].
    # We use a difference array.
    diff = [0] * (N + 2)
    # Use map/reduce to apply updates
    # For each j (0 to N-1), the range of i (1-based) is L_vals[j] to j+1.
    # Note: the condition is i < j. So i can be L_vals[j] up to j.
    # Example: H = [2, 1, 4, 3, 5], N=5
    # j=0 (H=2): L=1. i in [1, 0] -> None
    # j=1 (H=1): L=1. i in [1, 1] -> i=1
    # j=2 (H=4): L=1. i in [1, 2] -> i=1, 2
    # j=3 (H=3): L=3. i in [3, 3] -> i=3 (Wait, L_vals[3] is index of H=4, which is 2. 
    # So i is in [3, 3]? No. Let's re-evaluate.)
    
    # Correct logic:
    # j is visible from i if max(H[i+1...j-1]) < H[j].
    # This means i must be >= the index of the first building to the left of j 
    # that is taller than H[j].
    # Let's use 0-indexing for H.
    # For a fixed j, the condition holds for i < j such that 
    # for all k: i < k < j, H[k] < H[j].
    # This is true if i >= index of the nearest building to the left of j 
    # that is taller than H[j].
    # Let prev_greater[j] = max({k | k < j and H[k] > H[j]} union {-1})
    # Then the condition holds for i in {prev_greater[j], ..., j-1} 
    # EXCEPT that i cannot be less than 0.
    # Also, the problem says i is 1-indexed, and we need i < j.
    # Let's use 0-indexing for everything internally.
    # i ranges from 0 to N-1. We need j > i.
    # For a fixed j, the valid i's are i such that 0 <= i < j 
    # and for all k in (i, j), H[k] < H[j].
    # This means i >= prev_greater[j].
    # So i is in the range [max(0, prev_greater[j]), j-1].
    
    # Example 1: 2 1 4 3 5
    # j=0: H=2. prev=-1. i in [0, -1] -> None
    # j=1: H=1. prev=0. i in [0, 0] -> i=0
    # j=2: H=4. prev=-1. i in [0, 1] -> i=0, 1
    # j=3: H=3. prev=2. i in [2, 2] -> i=2
    # j=4: H=5. prev=-1. i in [0, 3] -> i=