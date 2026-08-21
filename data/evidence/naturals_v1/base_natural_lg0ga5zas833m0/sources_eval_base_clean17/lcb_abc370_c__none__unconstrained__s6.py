import sys

def solve():
    # Read S and T from standard input
    try:
        input_data = sys.stdin.read().split()
        if not input_data:
            return
        s = list(input_data[0])
        t = list(input_data[1])
    except EOFError:
        return

    n = len(s)
    
    # Identify indices where S and T differ
    diff_indices = []
    for i in range(n):
        if s[i] != t[i]:
            diff_indices.append(i)
    
    # The minimum number of operations is the number of differing characters
    m = len(diff_indices)
    print(m)
    
    if m == 0:
        return

    # To make the array X lexicographically smallest, we need to consider
    # the effect of changing S[i] to T[i].
    # If T[i] < S[i], changing it makes the string lexicographically smaller.
    # If T[i] > S[i], changing it makes the string lexicographically larger.
    # To keep the sequence of strings X lexicographically smallest, we should:
    # 1. First process all indices i where T[i] < S[i] in increasing order of i.
    # 2. Then process all indices i where T[i] > S[i] in decreasing order of i.
    
    smaller = [] # indices where t[i] < s[i]
    larger = []  # indices where t[i] > s[i]
    
    for i in diff_indices:
        if t[i] < s[i]:
            smaller.append(i)
        else:
            larger.append(i)
            
    # Sort 'smaller' indices ascending to reduce the string as early as possible
    smaller.sort()
    # Sort 'larger' indices descending to delay the increase of the string as long as possible
    larger.sort(reverse=True)
    
    # Combine the order of operations
    order = smaller + larger
    
    current_s = s
    for idx in order:
        current_s[idx] = t[idx]
        print("".join(current_s))

if __name__ == "__main__":
    solve()