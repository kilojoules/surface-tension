import sys

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    s = input_data[0]
    t = input_data[1]
    
    # Identify indices where S and T differ
    diff_indices = [i for i in range(len(s)) if s[i] != t[i]]
    
    # To minimize the number of elements, we must change exactly one character per step.
    # The number of elements M is the number of differing characters.
    # To make the array X lexicographically smallest, we must prioritize 
    # changing characters at indices that make the resulting string smaller.
    # A change at index i makes the string smaller if T[i] < S[i].
    # We should process all indices where T[i] < S[i] in increasing order of i,
    # then all indices where T[i] > S[i] in decreasing order of i.
    
    decreasing = [i for i in diff_indices if t[i] < s[i]]
    increasing = [i for i in diff_indices if t[i] > s[i]]
    
    # Sort decreasing indices ascending (to affect the leftmost character first)
    # Sort increasing indices descending (to affect the rightmost character first,
    # because changing a left character to something larger makes the string 
    # lexicographically larger immediately).
    
    # Wait, the logic for lexicographical smallest array:
    # 1. If we can make the string smaller, do it as far left as possible.
    # 2. If we must make the string larger, do it as far right as possible.
    
    # Correct order of indices to change:
    # First: indices i where T[i] < S[i], sorted ascending.
    # Second: indices i where T[i] > S[i], sorted descending.
    
    order = sorted(decreasing) + sorted(increasing, reverse=True)
    
    # Generate the sequence of strings
    s_list = list(s)
    x = [ "".join(s_list) for i in order if (s_list.__setitem__(i, t[i]) or True) ]
    
    # The list comprehension above is a trick to perform the mutation and capture the state.
    # Let's rewrite it cleanly.
    
    # Re-calculating X without the comprehension trick for clarity:
    s_list = list(s)
    x = []
    for i in order:
        s_list[i] = t[i]
        x.append("".join(s_list))
        
    # Output the results
    print(len(x))
    for string in x:
        print(string)

if __name__ == "__main__":
    solve()