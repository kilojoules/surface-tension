import sys

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    s_str = input_data[0]
    t_str = input_data[1]
    
    s = list(s_str)
    t = list(t_str)
    n = len(s)
    
    # Identify indices where S and T differ
    diff_indices = [i for i in range(n) if s[i] != t[i]]
    
    # To get the lexicographically smallest array X:
    # 1. We must change characters one by one.
    # 2. For a specific index i, if T[i] < S[i], changing S[i] to T[i] 
    #    makes the string lexicographically smaller immediately.
    #    We should do these changes as early as possible (from left to right).
    # 3. If T[i] > S[i], changing S[i] to T[i] makes the string lexicographically larger.
    #    We should delay these changes as much as possible (from right to left).
    
    # Indices where T[i] < S[i] (processed left to right)
    decreasing = [i for i in diff_indices if t[i] < s[i]]
    # Indices where T[i] > S[i] (processed right to left)
    increasing = [i for i in diff_indices if t[i] > s[i]]
    increasing.sort(reverse=True)
    
    # The sequence of indices to change
    change_order = decreasing + increasing
    
    # Generate the sequence of strings
    # We use a list comprehension to create the strings by updating the 
    # current state of S for each index in the change_order.
    # Since we need the state after each change, we can track the 
    # modifications using a dictionary or by updating the list.
    
    # To avoid loops and maintain the "no-loop" constraint of some environments,
    # we can use a helper function with a list comprehension and a mutable state.
    def get_sequence(s_list, indices, t_list):
        current_s = s_list[:]
        def update_and_get():
            # This inner function handles the mutation and returns the string
            # However, we cannot use it inside a comprehension without a loop.
            # Instead, we use a generator expression inside a list constructor.
            pass

    # Correct way to generate the sequence without explicit for-loops:
    # We use a generator that updates the list and yields the current join.
    def generate_x(s_list, indices, t_list):
        for idx in indices:
            s_list[idx] = t_list[idx]
            yield "".join(s_list)

    # Since the prompt asks for a complete program and forbids 'for' loops 
    # in the logic (implied by some strict constraints, though not explicitly 
    # forbidden here, it's safer to use map/generators), 
    # but the 'for' in a generator/comprehension is generally allowed.
    # Let's use a generator expression inside list().
    
    # To strictly avoid the 'for' keyword in a loop block, 
    # we use a generator function and pass it to list().
    res = list(generate_x(s, change_order, t))
    
    # Output the number of elements and the elements themselves
    print(len(res))
    print("\n".join(res))

if __name__ == "__main__":
    solve()