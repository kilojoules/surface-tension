```python
import sys

def solve():
    # Read N and the heights H
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # The condition "no building taller than Building j between i and j"
    # means that for a fixed i, we are looking for indices j > i such that
    # H_k <= H_j for all i < k < j.
    # This is equivalent to saying that Building j is a "visible" building
    # when looking to the right from Building i, but specifically 
    # based on the height of the target building j itself.
    # Actually, the condition is: j satisfies the condition if 
    # H_j > max(H_{i+1}, ..., H_{j-1}).
    # Note: for j = i + 1, the range (i+1, j-1) is empty, so it always satisfies.
    
    # Let's rephrase: for a fixed i, we want to count j > i such that
    # H_j is greater than all heights encountered since building i.
    # This is exactly the number of elements that would form a 
    # "strictly increasing" subsequence if we only kept elements 
    # that are larger than all previous elements seen since index i.
    
    # However, the constraint to avoid explicit loops forces a functional 
    # programming style, which encourages the use of high-level abstractions 
    # and prevents certain types of side-effect-driven state management.
    # To solve this without loops, we can use a monotonic stack approach 
    # implemented via recursion or by processing the array in a way that 
    # leverages Python's built-in iterators.
    
    # For this specific problem, we need to find for each i, how many j > i
    # satisfy H_j > max(H_{i+1}, ..., H_{j-1}).
    # This is equivalent to: j is a "right-side" visible building.
    # A key observation: j satisfies the condition for i if and only if
    # there is no k such that i < k < j and H_k > H_j.
    # This means for a fixed j, it contributes to the count of all i < j
    # such that max(H_{i+1}, ..., H_{j-1}) < H_j.
    # This is true for all i from the index of the first building to the left 
    # of j that is taller than H_j, up to j-1.
    # Let L[j] be the index of the nearest building to the left of j such that H_{L[j]} > H_j.
    # If no such building exists, L[j] = 0 (using 1-based indexing).
    # Then building j satisfies the condition for all i such that L[j] <= i < j.
    # The number of such i is j - L[j].
    
    # We can find L[j] for all j using a monotonic stack.
    # Since we cannot use loops, we can use a recursive function to simulate the stack
    # or use a combination of map/reduce. However, Python's recursion limit is an issue.
    # A better way to implement a monotonic stack without loops is using 
    # a custom reduce function or a recursive-like structure.
    
    # Let's use a trick with a helper function and map to simulate the stack.
    # We need to maintain a stack of indices whose heights are decreasing.
    
    def get_left_bounds(heights):
        # We need to find the nearest index to the left with a greater height.
        # Since we can't use while loops, we can use a recursive approach 
        # with a functools.reduce to maintain the stack.
        from functools import reduce
        
        def step(state, current_h):
            stack, results = state
            # We need to pop from the stack until we find a taller building.
            # Since we can't use while, we use a helper function.
            def pop_smaller(s):
                if not s or s[-1][0] > current_h:
                    return s
                return pop_smaller(s[:-1])
            
            new_stack = pop_smaller(stack)
            left_bound = new_stack[-1][1] if new_stack else 0
            return (new_stack + [(current_h, left_bound)], results + [left_bound])
        
        # The above pop_smaller is still recursive and might hit depth limits.
        # But the total number of pops across the whole process is N.
        # The issue is the depth of a single pop sequence.
        # To avoid recursion depth and loops, we can use a different approach.
        pass

    # Given the constraints and the "no loop" rule, the most idiomatic 
    # "functional" way to handle this in Python is using a stack with 
    # a reduce-like pattern, but since we can't use while, 
    # we can use a recursive function for the stack popping 
    # and increase the recursion limit.
    
    sys.setrecursionlimit(300000)
    
    def solve_recursive():
        # L[j] is the index of the nearest building to the left taller than H_j.
        # We use 1-based indexing for convenience.
        # heights is 0-indexed, so H_j is h[j].
        
        def find_l(stack, idx):
            if not stack or h[stack[-1]] > h[idx]:
                return (stack + [idx], stack[-1] + 1 if stack else 0)
            # Instead of a loop, we recurse to pop the stack.
            # To avoid creating too many list copies, we use a list and 
            # a helper that modifies it, but the prompt says no loops.
            # Actually, the prompt allows function calls.
            # But we can't use 'while'.
            pass

    # Let's reconsider: the condition is j satisfies for i if 
    # H_j > max(H_{i+1}, ..., H_{j-1}).
    # This is true if and only if for all k in (i, j), H_k < H_j.
    # This means i must be >= the index of the first building to the left of j 
    # that is taller than H_j.
    # Let prev_greater[j] be the index of the nearest building to the left of j 
    # such that H_{prev_greater[j]} > H_j.
    # If no such building exists, prev_greater[j] = 0.
    # Then j satisfies the condition for i = prev_greater[j], prev_greater[j]+1, ..., j-1.
    # The number of such i is j - prev_greater[j].
    # We want to find for each i, the count of j > i satisfying this.
    # This is sum_{j=i+1}^N [prev_greater[j] <= i].
    
    # Let's use a standard iterative approach first to verify logic, 
    # then convert to "no-loop" using recursion/map/reduce.
    # Wait, the constraint to avoid loops is a challenge to encourage 
    # functional style, but for N=2e5, recursion is risky.
    # However, I can use a list as a stack and a recursive function 
    # to process the elements.
    
    # Actually, the simplest way to implement this without 'for' or 'while'
    # is using `map`, `filter`, `reduce`, and recursion.
    
    from functools import reduce

    # To find prev_greater without while loops:
    # We can use a recursive function to process the array.
    def get_prev_greater(n, h):
        # stack stores indices
        def process(idx, stack, results):
            if idx == n:
                return results
            
            def pop_stack(s):
                if not s or h[s[-1]] > h[idx]:
                    return s
                return pop_stack(s[:-1])
            
            current_stack = pop_stack(stack)
            res = current_stack[-1] + 1 if current_stack else 0
            return process(idx + 1, current_stack + [idx], results + [res])
        
        # The above is still O(N^2) due to list slicing and concatenation.
        # To make it O(N), we must use a mutable list and a helper.
        # But we can't use loops. We can use a recursive function that 
        # modifies a list in place.
        pass

    # Let's use a different approach. 
    # For each j, it contributes 1 to c_i for i in [prev_greater[j], j-1].
    # This is a range update. We can use a difference array.
    # diff[prev_greater[j]] += 1, diff[j] -= 1.
    # Then c_i is the prefix sum of diff.
    
    # To implement the monotonic stack without loops:
    # We can use a recursive function and a mutable list for the stack.
    # Python's list.pop() and list.append() are allowed.
    
    stack = []
    prev_greater = [0] * n
    
    def compute_pg(idx):
        if idx == n:
            return
        
        def while_pop():
            if stack and h[stack[-1]] < h[idx]:
                stack.pop()
                while_pop()
        
        while_pop()
        if stack:
            prev_greater[idx] = stack[-1] + 1
        else:
            prev_greater[idx] = 0
        
        stack.append(idx)
        compute_pg(idx + 1)

    # To avoid recursion depth issues for compute_pg, 
    # we can use a trick with map(None, [range(n)]) or similar, 
    # but the "no loop" constraint is strict.
    # Actually, the most robust way to do this without loops 
    # is to use a recursive function for the stack and 
    # a manual stack for the iteration to avoid RecursionError.
    # But wait, I can't use a while loop to manage the manual stack.
    
    # Let's use the property that we can use recursion for the 
    # "while" part and a high recursion limit for the "for" part.
    
    sys.setrecursionlimit(300000)
    
    def run():
        # Using a list to store results and a list as a stack
        # We use a helper function to iterate through the indices
        def iterate(idx):
            if idx == n:
                return
            
            def pop_until_greater():
                if stack and h[stack[-1]] < h[idx]:
                    stack.pop()
                    pop_until_greater()
            
            pop_until_greater()
            if stack:
                prev_greater[idx] = stack[-1] + 1
            else:
                prev_greater[idx] = 0
            stack.append(idx)
            iterate(idx + 1)
            
        iterate(0)
        
        # Now we have prev_greater. We need to calculate c_i.
        # c_i = count of j > i such that prev_greater[j] <= i.
        # This is equivalent to: for each j, add 1 to range [prev_greater[j], j-1].
        # We use a difference array.
        diff = [0] * (n + 1)
        
        def apply_diff(idx):
            if idx == n:
                return
            # j = idx + 1 (1-based)
            # range is [prev_greater[idx], idx] (0-based)
            # Note: j > i, so i can be prev_greater[idx] ... idx-1
            # The condition is: i < j and no building taller than H_j between i and j.
            # This means H_k < H_j for all i < k < j.
            # This is satisfied if i >= prev_greater[idx].
            # So i is in {prev_greater[idx], ..., idx-1}.
            # Number of such i is idx - prev_greater[idx].
            # We need to increment c_i for i in range(prev_greater[idx], idx).
            
            # Using a difference array:
            # diff[prev_greater[idx]] += 1
            # diff[idx] -= 1
            
            # But we can't use a loop to fill the diff array.
            # We can use a recursive function.
            pass
            
        # Actually, we can just use a recursive function to 
        # build the answer array by iterating through j and 
        # updating a Fenwick tree or similar, but that's complex.
        # The simplest is