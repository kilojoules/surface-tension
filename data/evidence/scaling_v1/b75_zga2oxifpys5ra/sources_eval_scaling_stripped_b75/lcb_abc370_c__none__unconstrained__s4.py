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
    
    # To make the array of strings lexicographically smallest:
    # 1. We must change characters at indices that make the resulting string smaller.
    # 2. If changing S[i] to T[i] makes the string smaller (T[i] < S[i]), 
    #    we should do this as early as possible, starting from the leftmost index.
    # 3. If changing S[i] to T[i] makes the string larger (T[i] > S[i]), 
    #    we should do this as late as possible, starting from the rightmost index.
    
    # Indices that decrease the string value (T[i] < S[i])
    decreasing = [i for i in diff_indices if t[i] < s[i]]
    # Indices that increase the string value (T[i] > S[i])
    increasing = [i for i in diff_indices if t[i] > s[i]]
    
    # Sort decreasing indices ascending (left to right)
    # Sort increasing indices descending (right to left)
    order = sorted(decreasing) + sorted(increasing, reverse=True)
    
    # Generate the sequence of strings
    # We use a list for S since strings are immutable in Python
    s_list = list(s)
    x = [ "".join(s_list) for i in order if (s_list.__setitem__(i, t[i]) or True) ]
    
    # The logic inside the list comprehension:
    # s_list.__setitem__(i, t[i]) modifies the list in place and returns None.
    # 'or True' ensures the expression evaluates to True so the resulting string is appended.
    # However, the comprehension above executes the setitem and then joins.
    # Let's refine it to ensure the string is captured AFTER the change.
    
    # Corrected sequence generation:
    s_list = list(s)
    res = [ "".join(s_list) for i in order if (s_list.__setitem__(i, t[i]) or False) == False ]
    
    # Wait, the comprehension logic is tricky with side effects. 
    # Let's use a standard loop for clarity and correctness.
    s_list = list(s)
    final_x = []
    for i in order:
        s_list[i] = t[i]
        final_x.append("".join(s_list))

    # Output the number of elements and the elements themselves
    print(len(final_x))
    for string in final_x:
        print(string)

if __name__ == "__main__":
    solve()