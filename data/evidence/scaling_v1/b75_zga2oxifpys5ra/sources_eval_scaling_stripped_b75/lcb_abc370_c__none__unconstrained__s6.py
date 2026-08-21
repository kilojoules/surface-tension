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
    
    # To minimize the number of elements, we must change exactly one 
    # differing character per step. Total steps M = number of differences.
    # To make the array X lexicographically smallest, we must prioritize
    # changes that make the string lexicographically smaller.
    # 1. If s[i] > t[i], changing s[i] to t[i] makes the string smaller.
    #    We should do these as early as possible, from left to right.
    # 2. If s[i] < t[i], changing s[i] to t[i] makes the string larger.
    #    We should do these as late as possible, from right to left.
    
    decreasing = [i for i in diffs if s[i] > t[i]]
    increasing = [i for i in diffs if s[i] < t[i]]
    
    # Sort decreasing indices ascending (left to right)
    # Sort increasing indices descending (right to left)
    order = sorted(decreasing) + sorted(increasing, reverse=True)
    
    # Generate the sequence of strings
    # We use a list comprehension to create the strings by updating the 
    # current state of s. Since we cannot use loops, we track the 
    # state by mapping the index to the target character.
    
    # For each step k in 0...M-1, the character at order[k] is changed to t[order[k]].
    # A character at index i is t[i] if i has appeared in order[0...k].
    
    def get_string_at_step(k):
        # Indices changed up to step k
        changed_indices = set(order[:k+1])
        return "".join([t[i] if i in changed_indices else s[i] for i in range(n)])

    # The number of elements M
    m = len(order)
    
    # Generate the array X
    x = [get_string_at_step(k) for k in range(m)]
    
    # Output M and then each string in X
    print(m)
    print("\n".join(x))

if __name__ == "__main__":
    solve()