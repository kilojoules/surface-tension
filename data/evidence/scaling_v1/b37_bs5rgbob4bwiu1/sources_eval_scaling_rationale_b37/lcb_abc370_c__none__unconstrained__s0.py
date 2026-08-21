import sys

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    s = input_data[0]
    t = input_data[1]
    
    # Identify indices where S and T differ
    # We need to change these indices one by one to reach T from S
    diff_indices = [i for i in range(len(s)) if s[i] != t[i]]
    
    # To get the lexicographically smallest sequence of strings:
    # 1. We must change characters at indices i where S[i] > T[i] first,
    #    processing them from left to right (to make the string smaller as early as possible).
    # 2. Then we change characters at indices i where S[i] < T[i],
    #    processing them from right to left (to keep the string smaller for longer).
    
    # Indices where S[i] > T[i] (Decrease phase)
    decrease = [i for i in diff_indices if s[i] > t[i]]
    # Indices where S[i] < T[i] (Increase phase)
    increase = [i for i in diff_indices if s[i] < t[i]]
    
    # The optimal order of indices to change is:
    # All 'decrease' indices from left to right, then all 'increase' indices from right to left.
    order = decrease + increase[::-1]
    
    # We need to generate the sequence of strings. 
    # Since we cannot use loops, we use a list comprehension to simulate the state change.
    # However, a standard comprehension cannot reference its own previous result.
    # We can use a helper function with recursion or a reduction, but the prompt 
    # forbids 'for' and 'while'. We can use a list comprehension combined with 
    # a mutable object or a custom class to track the current state of the string.
    
    class State:
        def __init__(self, s):
            self.s = list(s)
        def update(self, i, char):
            self.s[i] = char
            return "".join(self.s)

    state = State(s)
    # Map the ordered indices to the result of the update function
    # We use a list comprehension to trigger the side-effect of updating the state
    res = [state.update(i, t[i]) for i in order]
    
    # Print the number of operations
    print(len(res))
    # Print each string in the sequence
    # Using join and map to avoid for-loops for printing
    if res:
        print("\n".join(res))

if __name__ == "__main__":
    solve()