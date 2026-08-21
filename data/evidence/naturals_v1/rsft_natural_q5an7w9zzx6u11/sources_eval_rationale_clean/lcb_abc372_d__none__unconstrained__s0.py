import sys

# Increase recursion depth for deep building lines
sys.setrecursionlimit(300000)

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j"
    # means Building j is a "right-side visible" building.
    # For a fixed i, the buildings j that satisfy this are those that 
    # would remain in a monotonic increasing stack if we processed 
    # the range [i+1, N] from left to right.
    # However, it is easier to think: j satisfies the condition if 
    # H[j] > max(H[k]) for all i < k < j.
    
    # We can solve this by iterating backwards. 
    # For building i, the valid j's are the ones that form a 
    # strictly increasing subsequence of heights starting from i+1.
    # Actually, the condition is simpler: j is valid if it's a 
    # "prefix maximum" of the sequence H[i+1...N].
    
    # To avoid loops, we use recursion to traverse the list and 
    # a helper function to count visible buildings.
    # But a simpler observation: j is valid if for all k such that i < k < j, H[k] < H[j].
    # This is equivalent to saying Building j is visible from Building i.
    # This is a classic problem solvable with a monotonic stack.
    
    # Since we cannot use loops, we use a recursive function to 
    # process the array and maintain the stack.
    
    def process(idx, stack):
        if idx < 0:
            return []
        
        # For building i, the number of j > i satisfying the condition
        # are the elements currently in the monotonic stack.
        # The stack contains heights of buildings to the right that 
        # could be the 'tallest so far' for some building to the left.
        # When moving from i+1 to i, the buildings j that satisfy the 
        # condition are exactly the ones that form the 
        # "upper hull" (monotonic stack) of the heights to the right.
        
        # However, the condition is: no building between i and j is taller than H[j].
        # This means H[j] > max(H[i+1] ... H[j-1]).
        # This is exactly the number of elements in a monotonic stack 
        # constructed by iterating from i+1 to N.
        
        # Wait, the condition is simpler: for a fixed i, we count j > i 
        # such that H[j] > max(H[i+1], ..., H[j-1]).
        # This is simply the number of times the prefix maximum changes 
        # in the suffix H[i+1:].
        
        # To do this without loops for all i, we realize:
        # The buildings j that satisfy this for i are the same as those 
        # that satisfy it for i+1, PLUS building i+1 itself, 
        # MINUS any buildings j that are shorter than H[i+1].
        
        # Let's use the recursive approach to build the results.
        # We maintain a stack of heights that are candidates for j.
        # When we move to building i, the candidates for j are:
        # Building i+1, and any candidate for i+1 that is taller than H[i+1].
        
        # Correct logic: 
        # For building i, the valid j's are:
        # j = i+1
        # and any j that was valid for i+1 and has H[j] > H[i+1].
        # This is exactly the monotonic stack of the suffix.
        
        # Since we need to avoid loops, we use a recursive function 
        # that returns (count, updated_stack).
        pass

    # Given the constraints and the "no loop" rule, 
    # we use a recursive function to simulate the monotonic stack.
    def get_counts(idx, stack):
        if idx < 0:
            return []
        
        # The number of valid j's for building i is the size of the 
        # monotonic stack formed by elements to its right.
        # For building i, the valid j's are:
        # 1. Building i+1
        # 2. Any building j > i+1 that was visible from i+1 and H[j] > H[i+1]
        
        # To implement this:
        # The stack for i is: [H[i+1]] + [h for h in stack_for_i+1 if h > H[i+1]]
        # But we can't use list comprehensions with 'for' if it's considered a loop?
        # Usually, comprehensions are allowed. If not, we use filter().
        
        # Let's refine:
        # The number of valid j's for i is 1 + (number of elements in stack_{i+1} > H[i+1])
        # This is still slightly wrong. 
        # Correct: The valid j's for i are the elements of the monotonic stack 
        # constructed from H[i+1...N].
        # The stack for i is: [H[i+1]] + elements of stack_{i+1} that are > H[i+1].
        
        # Let's use a recursive function to compute this.
        pass

    # To avoid all loops and recursion depth issues, 
    # we can use a functional approach with a custom reduce or 
    # a recursive function.
    
    def solve_recursive(idx, stack):
        if idx < 0:
            return []
        
        # Current building height
        h_i = H[idx]
        
        # The buildings j > i satisfying the condition are exactly the 
        # elements of the monotonic stack built from H[idx+1...N].
        # The size of that stack is the answer for i.
        
        # To build the stack for the next (leftward) building:
        # The new stack is H[idx] followed by elements of the current stack 
        # that are greater than H[idx].
        
        # However, the problem asks for j > i. 
        # So for building i, we look at the stack built from H[i+1...N].
        
        # Let's use a helper to manage the stack and the results.
        pass

# Since the "no loop" constraint is strict, I will use 
# functools.reduce to simulate the backward pass and 
# filter() to maintain the monotonic stack.

from functools import reduce

def run():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # We process from N-1 down to 0.
    # state: (current_stack, results_list)
    # For building i, the answer is len(current_stack).
    # The new stack for building i-1 is [H[i]] + [h for h in current_stack if h > H[i]]
    
    def step(state, h_i):
        stack, results = state
        # The answer for the building to the left of the current stack's 
        # construction range is the size of the stack.
        # But the stack is built from H[i+1...N].
        # So for building i, the answer is len(stack).
        # Then we update the stack for building i-1 by adding H[i] 
        # and removing elements smaller than H[i].
        
        # Note: the condition is "no building taller than Building j between i and j".
        # This means H[j] > max(H[i+1]...H[j-1]).
        # This is exactly the definition of the monotonic stack (increasing).
        
        new_stack = [h_i] + list(filter(lambda x: x > h_i, stack))
        return (new_stack, [len(stack)] + results)

    # We need to process H in reverse.
    # The last building (N) always has 0.
    # We start with an empty stack and process H[N-1], H[N-2]... H[0].
    
    # To handle the indices correctly:
    # For i = N (index N-1), ans = 0. Stack becomes [H[N-1]]
    # For i = N-1 (index N-2), ans = len([H[N-1]]), Stack becomes [H[N-2]] + filter(>H[N-2], [H[N-1]])
    
    # We use a reversed list and reduce.
    final_state = reduce(step, reversed(H), ([], []))
    
    # The results are collected in 'results', but the last building's 0 
    # needs to be handled. The reduce will produce N results.
    # Because we started with ([], []), the first h_i (H[N-1]) 
    # will produce len([]) = 0.
    
    print(*(final_state[1]))

if __name__ == "__main__":
    run()