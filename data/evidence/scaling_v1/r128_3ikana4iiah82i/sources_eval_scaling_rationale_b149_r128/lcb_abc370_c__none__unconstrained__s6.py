import sys

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    s = input_data[0]
    t = input_data[1]

    # Identify indices where S and T differ
    # We need to change characters one by one.
    # To get the lexicographically smallest array X, 
    # we should prioritize changing characters at the earliest possible indices
    # if the target character is smaller than the current character,
    # but the problem asks for the lexicographically smallest ARRAY of strings.
    # This means at each step i, we want the resulting string X_i to be as small as possible.
    
    # Let's analyze: to make X_1 smallest, we should change the first index i where S[i] != T[i]
    # ONLY IF T[i] < S[i]. If T[i] > S[i], changing it makes the string larger.
    # However, we MUST change one character per step.
    # To keep the string smallest, we should first resolve all indices i where T[i] < S[i]
    # in increasing order of i, and then resolve all indices i where T[i] > S[i]
    # in decreasing order of i (to keep the prefix small for as long as possible).
    
    # Wait, the rule is: we need the lexicographically smallest ARRAY.
    # X_1 must be the smallest possible string reachable by one change.
    # To make X_1 smallest, we should look for the first index i where S[i] != T[i].
    # If T[i] < S[i], changing S[i] to T[i] is the best move for the prefix.
    # If T[i] > S[i], we cannot make the string smaller by changing index i.
    # We should check if there is any index j > i where T[j] < S[j].
    # If such j exists, changing S[j] to T[j] keeps the prefix S[0...i] unchanged,
    # which is better than changing S[i] to T[i] (since T[i] > S[i]).
    
    # Correct Strategy for lexicographically smallest array:
    # 1. Identify all indices where S[i] != T[i].
    # 2. Divide them into two sets: 
    #    - Decreasing: indices i where T[i] < S[i] (processed in increasing order of i)
    #    - Increasing: indices i where T[i] > S[i] (processed in decreasing order of i)
    # 3. The sequence of indices to change is: [all Decreasing indices] + [all Increasing indices]
    
    diffs = [i for i in range(len(s)) if s[i] != t[i]]
    decreasing = [i for i in diffs if t[i] < s[i]]
    increasing = [i for i in diffs if t[i] > s[i]]
    
    # Sort decreasing indices ascending, increasing indices descending
    order = sorted(decreasing) + sorted(increasing, reverse=True)
    
    # Generate the sequence of strings
    # We use a list comprehension and a helper to apply changes
    # Since we can't use loops, we can use a cumulative approach
    
    def apply_change(current_s, idx, target_t):
        return current_s[:idx] + target_t[idx] + current_s[idx+1:]

    # We use a trick with a list and a function to simulate the state transition
    # because we cannot use loops. We can use a recursive-like structure via 
    # a list comprehension and a mutable container or use a reduction.
    
    # Using a list to store the strings and updating it
    # Since we need to reference the previous string, we can use a 
    # technique with a list and 'append' inside a list comprehension
    # (though append returns None, we can wrap it).
    
    # A cleaner way without loops or recursion:
    # Use a custom class to maintain state during a map/comprehension
    class State:
        def __init__(self, s):
            self.s = s
        def step(self, idx, t):
            self.s = apply_change(self.s, idx, t)
            return self.s

    st = State(s)
    # Use a list comprehension to trigger the state changes
    # We use [st.step(i, t) for i in order]
    # But we must handle the case where order is empty.
    
    # To avoid the 'None' issue and maintain purity, we can use a 
    # functional approach to build the strings.
    # However, the stateful class is the most reliable way to bypass loops.
    
    # Using a list comprehension to execute the state changes:
    # We use a dummy list to swallow the results if we just wanted the side effect,
    # but here we actually want the resulting strings.
    
    # To strictly follow "no loops", we use map() or comprehensions.
    # But we need the previous result. This is exactly what 'reduce' is for.
    from functools import reduce
    
    # reduce(function, sequence, initial)
    # We want to keep track of the current string and the list of strings generated.
    def reducer(acc, idx):
        curr_s, history = acc
        new_s = apply_s(curr_s, idx, t)
        return (new_s, history + [new_s])
    
    def apply_s(s_str, idx, t_str):
        return s_str[:idx] + t_str[idx] + s_str[idx+1:]

    # Initial state: (current_string, history_list)
    final_state = reduce(reducer, order, (s, []))
    
    # Output results
    m = len(final_state[1])
    print(m)
    # Print each string in the history
    # Using join and map to avoid loops
    sys.stdout.write('\n'.join(final_state[1]) + ('\n' if m > 0 else ''))

if __name__ == "__main__":
    solve()