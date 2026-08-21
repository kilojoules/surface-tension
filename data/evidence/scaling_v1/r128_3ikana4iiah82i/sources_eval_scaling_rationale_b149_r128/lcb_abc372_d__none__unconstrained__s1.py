import sys
from itertools import accumulate

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j"
    # means we are looking for elements to the right of i that are 
    # "visible" if we look from i.
    # Specifically, for a fixed i, we want to count j > i such that
    # H_j > max(H_{i+1}, ..., H_{j-1}).
    # This is equivalent to counting elements in a monotonic increasing stack
    # when traversing from i+1 to N.
    # However, it is easier to think about this as: for a fixed i,
    # we want to count how many j > i are "prefix maximums" of the suffix starting at i+1.
    
    # To solve this for all i efficiently, we can use a stack-based approach.
    # When moving from right to left, we maintain a stack of indices of buildings
    # that could be the "tallest" for some i to the left.
    # The stack will store indices j such that H_j is strictly increasing from top to bottom.
    
    # We use accumulate to simulate a stack. 
    # The state is (stack, count_for_current_i).
    # For the current building i, the number of j's is the size of the stack 
    # after we remove all buildings from the top of the stack that are shorter than H_i.
    # Wait, the condition is: no building taller than H_j between i and j.
    # This means H_j must be a record-breaker (prefix maximum) starting from i+1.
    
    # Correct logic:
    # For a fixed i, we are looking for j > i such that H_j > max(H_{i+1} ... H_{j-1}).
    # This is exactly the number of elements that would remain in a monotonic 
    # decreasing stack (of heights) if we processed the suffix from i+1 to N.
    # Actually, the number of such j is simply the number of elements in the 
    # monotonic stack of the suffix [i+1, N].
    
    # Let's use accumulate to build the stack from right to left.
    # For index i, we want to know the size of the monotonic stack of the suffix [i+1, N].
    # But the stack depends on the values. 
    # Let's redefine: for index i, the valid j's are the ones that form the 
    # "upper hull" of the heights to the right.
    # If we process from right to left, and we are at index i, 
    # the valid j's are the ones in the stack where we keep elements 
    # that are taller than everything to their right.
    
    # Let's refine: 
    # For i, we want j > i such that H_j > max(H_{i+1}, ..., H_{j-1}).
    # This means H_{i+1} is always a candidate (j=i+1).
    # Then we look for the first k > i+1 such that H_k > H_{i+1}, and so on.
    # This is exactly the number of elements in a monotonic stack 
    # constructed by iterating from i+1 to N.
    
    # To do this for all i, we can observe that the "visible" buildings from i
    # are the same as the "visible" buildings from i+1, PLUS building i+1 itself,
    # MINUS any buildings that were visible from i+1 but are shorter than H_{i+1}.
    # Actually, the set of j's for i is: {i+1} union {j > i+1 | H_j > H_{i+1} and j was visible from i+1}.
    
    # Let's use a stack and process from right to left.
    # For index i, the answer is the size of the stack after we push H_{i+1} 
    # and remove all elements smaller than H_{i+1} from the top.
    
    # We can use accumulate to maintain the stack.
    # The state: (stack, current_ans)
    # We process the array in reverse.
    
    def step(state, height):
        stack, _ = state
        # Remove elements from stack that are smaller than current height
        # Since we can't use while loops, we use a recursive-like structure 
        # via a helper function or a clever trick.
        # But the constraint says no recursion.
        # We can use a list comprehension to filter the stack, but that's O(N).
        # Wait, the "no loop" constraint is tricky for stacks.
        # Let's use the property: the answer for i is 1 + (ans for i+1 if H_{i+1} is the max)
        # Actually, the simplest way to implement a monotonic stack without loops 
        # is to use a function that handles the popping.
        # Since I can't use while, I'll use a trick with `functools.reduce` 
        # and a helper that manages the stack.
        pass

    # Re-evaluating: The condition "no building taller than H_j between i and j"
    # means H_j is a prefix maximum of the sequence H_{i+1}, ..., H_N.
    # The number of prefix maximums of a sequence is the size of the 
    # monotonic stack after processing the sequence.
    
    # To avoid loops/recursion, I will use a technique with 
    # a custom class or a closure to maintain state within 
    # a list comprehension or map.
    
    class StackManager:
        def __init__(self, heights):
            self.h = heights
            self.stack = []
            self.results = []
        
        def process(self, i):
            # We are processing from right to left.
            # For index i, we want to find the number of prefix maximums of H[i+1...N-1]
            # The stack contains the prefix maximums of the suffix processed so far.
            # When we move from i+1 to i, we are adding H[i+1] to the front of the suffix.
            # The new prefix maximums are: H[i+1] and all previous prefix maximums that are > H[i+1].
            
            # Since we are going right to left, the 'suffix' is growing to the left.
            # Let's maintain a stack of the suffix's prefix maximums.
            # When we add H[i+1], we pop all elements from the stack smaller than H[i+1].
            # The size of the stack is the answer for i.
            
            # To avoid 'while', we can use a recursive-like structure 
            # using a list comprehension and a slice, but that's O(N^2).
            # However, we can use a trick: 
            # The number of elements to pop is the number of elements in the stack 
            # smaller than H[i+1]. 
            # We can find this using bisect if the stack is sorted.
            # The stack of prefix maximums (processed right to left) will be 
            # sorted if we store them correctly.
            pass

    # Correct approach:
    # Process from right to left. Maintain a stack of heights that are 
    # "visible" from the left.
    # For index i, the answer is the size of the stack after 
    # pushing H[i+1] and popping all elements < H[i+1].
    # Since the stack is always sorted (bottom to top: decreasing), 
    # we can use binary search to find how many elements to pop.
    
    import bisect
    
    # We need to maintain the stack and the answers.
    # We can use accumulate to pass the stack along.
    def manage_stack(state, height):
        stack = state
        # The stack stores heights of visible buildings from the current i.
        # It is sorted in increasing order from top to bottom (bottom is tallest).
        # When we add a new building (which is to the left of the current stack),
        # it becomes the first visible building. 
        # Any building in the stack shorter than it is no longer visible.
        
        # Find index of first element >= height
        idx = bisect.bisect_left(stack, height)
        # The new stack is [height] + stack[idx:]
        return [height] + stack[idx:]

    # We process H from index N-1 down to 0.
    # For i = N-1, ans = 0.
    # For i = N-2, we process H[N-1], stack becomes [H[N-1]], ans = 1.
    # For i = N-3, we process H[N-2], stack becomes [H[N-2]] + {s in stack | s > H[N-2]}, ans = len(stack).
    
    # Use accumulate to build the stacks for each i.
    # We process the heights in reverse: H[N-1], H[N-2]... H[0]
    # The stack after processing H[i+1] gives the answer for i.
    
    # Note: H is 0-indexed. Building i (1-indexed) is H[i-1].
    # For i=N, ans=0.
    # For i=N-1, we process H[N-1], stack is [H[N-1]], ans=1.
    # ...
    # For i=1, we process H[1...N-1].
    
    # We need the heights from index 1 to N-1 in reverse order.
    heights_to_process = h[1:][::-1]
    
    # accumulate returns the state after each step.
    stacks = list(accumulate(heights_to_process, manage_stack, initial=[]))
    
    # The answers are the lengths of these stacks.
    # stacks[0] is for i=N (empty), stacks[1] is for i=N-1, etc.
    # We need them for i=1, 2, ..., N.
    # The accumulate result is in reverse order of i.
    ans = [len(s) for s in stacks][::-1]
    
    print(*(ans))

if __name__ == "__main__":
    solve()