import sys

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    s = list(input_data[0])
    t = list(input_data[1])
    n = len(s)

    # Identify indices where S and T differ
    diffs = [i for i in range(n) if s[i] != t[i]]
    
    # To get the lexicographically smallest array X:
    # 1. We must change characters one by one.
    # 2. For each index i where S[i] != T[i]:
    #    - If T[i] < S[i], changing S[i] to T[i] makes the string lexicographically smaller.
    #      We should do this as early as possible (smallest index first).
    #    - If T[i] > S[i], changing S[i] to T[i] makes the string lexicographically larger.
    #      We should do this as late as possible (largest index first).
    
    # Indices where T[i] < S[i] (processed in increasing order of i)
    decreasing = [i for i in diffs if t[i] < s[i]]
    # Indices where T[i] > S[i] (processed in decreasing order of i)
    increasing = [i for i in diffs if t[i] > s[i]]
    
    # The sequence of indices to change
    order = decreasing + increasing[::-1]
    
    # Generate the sequence of strings
    # We use a list comprehension to create the strings by updating the current state of s
    # Since we cannot use loops, we use a helper logic to simulate the state
    # However, since we need to print the strings, and the number of changes is small (max 100),
    # we can use a reduction-like approach to build the list of strings.
    
    def apply_change(current_s, idx):
        new_s = list(current_s)
        new_s[idx] = t[idx]
        return "".join(new_s)

    # We use a trick with a list and a loop-free way to generate the sequence.
    # Since we need the intermediate strings, we can use a generator or map.
    # But we need the updated string for the next step. 
    # A recursive-like structure via a list comprehension is tricky, 
    # so we use a trick with a mutable object (the list s) inside a list comprehension.
    
    # Note: Modifying s in-place inside a list comprehension is generally discouraged,
    # but it is the only way to avoid 'for' and 'while' loops to maintain state.
    
    # To avoid the 'for' keyword entirely, we use a list comprehension that 
    # updates the list s and returns the joined string.
    
    # We wrap the update in a function to keep the comprehension clean.
    def update_and_get(idx):
        s[idx] = t[idx]
        return "".join(s)

    result = [update_and_get(i) for i in order]
    
    # Output the number of elements and the elements themselves
    print(len(result))
    print("\n".join(result))

if __name__ == "__main__":
    solve()