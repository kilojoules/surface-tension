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
    
    # To get the lexicographically smallest array X:
    # 1. We must change characters one by one (minimum M is the number of differences).
    # 2. For each step, we want the resulting string to be as small as possible.
    # 3. If we change S[i] to T[i], and T[i] < S[i], we should do this as early as possible
    #    because it decreases the string lexicographically.
    # 4. If T[i] > S[i], we should do this as late as possible because it increases the string.
    
    # Strategy: 
    # First, process all indices i where T[i] < S[i] in increasing order of i.
    # Then, process all indices i where T[i] > S[i] in decreasing order of i.
    
    decreasing = [i for i in diff_indices if t[i] < s[i]]
    increasing = [i for i in diff_indices if t[i] > s[i]]
    
    # Sort decreasing indices ascending (to make the string smaller as early as possible)
    # Sort increasing indices descending (to delay making the string larger)
    order = sorted(decreasing) + sorted(increasing, reverse=True)
    
    # Generate the sequence of strings
    current_s = list(s)
    x = [
        "".join(current_s := [current_s[j] if j != i else t[j] for j in range(len(s))])
        for i in order
    ]
    
    # Output the number of elements and the elements themselves
    print(len(x))
    for string in x:
        print(string)

if __name__ == "__main__":
    solve()