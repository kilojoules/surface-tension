import sys
from functools import reduce

def solve():
    # Read N and the heights
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j"
    # means we are looking for the number of elements to the right of i
    # that would be visible if we were looking from building i.
    # Specifically, j satisfies the condition if H[k] < H[j] for all i < k < j.
    # This is equivalent to saying that j is a "right-side" visible building.
    # However, the problem asks for this for every i.
    # A simpler interpretation: for a fixed i, we are counting j > i such that
    # H[j] > max(H[i+1]...H[j-1]).
    # This means we are counting the elements of the upper-envelope/monotonic 
    # increasing subsequence starting from index i+1.
    
    # To solve this efficiently for all i, we process from right to left.
    # We maintain a monotonic stack of heights that could be "visible".
    # For a building i, the buildings j > i that satisfy the condition are
    # exactly the elements of the monotonic stack if we maintain it such that
    # we only keep elements that are taller than all elements to their left
    # (relative to the current i).
    
    # Actually, the condition is: j satisfies it if for all k where i < k < j, H[k] < H[j].
    # This means if we are at index i, we are looking for j > i such that 
    # H[j] is a left-to-right maximum of the suffix H[i+1:].
    # The number of such j is simply the size of the monotonic stack 
    # maintained by processing the array from right to left.
    # When moving from i+1 to i, we push H[i+1] onto the stack and 
    # remove all elements smaller than H[i+1] from the top because 
    # they will now be "blocked" by H[i+1] for any index <= i.
    
    # We use reduce to simulate the process:
    # State: (stack, results_list)
    # We process the heights in reverse order.
    
    def accumulate(state, h):
        stack, results = state
        # Remove elements from the stack that are smaller than the current height h
        # Since we need to remove elements from the end, we use a list and 
        # a custom logic. However, since we need the size of the stack 
        # AFTER pushing H[i+1], we handle the logic carefully.
        
        # For building i, we want to count j > i.
        # The buildings j that satisfy the condition are those that form 
        # a strictly increasing subsequence starting from the first element 
        # to the right of i.
        # Let's refine: for a fixed i, j=i+1 always satisfies it.
        # Then we look for the next j > i+1 such that H[j] > H[i+1], and so on.
        # This is exactly the size of the monotonic stack maintained by 
        # processing from right to left, where we pop elements smaller than 
        # the current element.
        
        # Because we need to avoid loops, we use a helper function or 
        # slice/comprehension to simulate the pop.
        # But wait, the "pop" depends on the value. A recursive-like 
        # structure is needed. Since we can't use recursion, we can 
        # use a trick with a while-like behavior inside a list comprehension 
        # or just use the fact that we can't use loops and must use 
        # functional tools.
        
        # Correct logic:
        # Process i from N-1 down to 1.
        # For i, the answer is the size of the monotonic stack of H[i+1...N].
        # The stack is maintained by: while stack and stack[-1] < H[i+1]: stack.pop()
        # Then stack.append(H[i+1]).
        pass

    # Since the constraint forbids loops and recursion, and we need to 
    # maintain a stack, the most reliable way to simulate the 'while' 
    # loop for popping is to use a helper function with a 
    # conditional expression and a list, but that requires recursion.
    # Alternatively, we can use a technique with `bisect` or 
    # other libraries, but the most "functional" way to handle 
    # the monotonic stack without loops is to use a 
    # custom class or a closure that manages state, 
    # but the prompt forbids loops entirely.
    
    # Let's use a different approach: 
    # The number of j's for index i is the number of elements in the 
    # monotonic stack after processing H[i+1].
    # We can use a helper function that uses a list as a stack and 
    # we use a trick to simulate the while loop using a 
    # generator or a complex reduce.
    
    # Actually, the most Pythonic way to avoid loops/recursion 
    # while maintaining state is to use a class and `map` or `reduce`.
    
    class StackManager:
        def __init__(self):
            self.stack = []
            self.results = []
        
        def push(self, h):
            # We need to remove elements smaller than h from the end.
            # Since we can't use while, we can use a slice and 
            # find the index where elements become >= h.
            # But we can't use a loop to find that index.
            # We can use a list comprehension to find all indices 
            # and then slice.
            # Wait, we can use a generator expression with `next` 
            # to find the first index from the right that is >= h.
            
            # Find index of last element >= h
            # We search backwards through the stack.
            idx = next((k for k in range(len(self.stack)-1, -1, -1) 
                        if self.stack[k] >= h), -1)
            self.stack = self.stack[:idx+1] + [h]
            # The answer for the building to the left of the one 
            # that pushed this h is the current size of the stack.
            # However, the building at index i is looking at the stack 
            # formed by H[i+1...N].
            return len(self.stack)

    # To avoid the 'for' loop in the class, we use map.
    # We process H in reverse. For i = N-1, N-2, ..., 0:
    # The answer for i is the size of the stack after processing H[i+1].
    # For i = N-1, answer is 0.
    # For i = N-2, we push H[N-1], answer is len(stack).
    # For i = N-3, we push H[N-2], answer is len(stack).
    
    sm = StackManager()
    # We need the results for i = 0, 1, ..., N-1.
    # For i = N-1, result is 0.
    # For i < N-1, result is the size of the stack after processing H[i+1...N-1].
    # We process H[N-1], H[N-2]... H[1].
    
    # Use map to process the heights in reverse order (from index N-1 down to 1)
    # The first element processed is H[N-1], which gives the result for i = N-2.
    res = [0] + list(map(sm.push, H[::-1][:-1]))
    
    # The results are generated in reverse order of i.
    # res[0] is for i=N-1, res[1] is for i=N-2, etc.
    # So we reverse it back.
    print(*(res[::-1]))

if __name__ == "__main__":
    solve()