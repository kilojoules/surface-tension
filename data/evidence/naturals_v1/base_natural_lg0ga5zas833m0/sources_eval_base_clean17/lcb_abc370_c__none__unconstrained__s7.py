import sys

def solve():
    # Read S and T from standard input
    try:
        s = sys.stdin.readline().strip()
        t = sys.stdin.readline().strip()
    except EOFError:
        return

    if not s or not t:
        # Handle potential empty input cases
        if s == t:
            print(0)
            return
        else:
            return

    n = len(s)
    # Identify positions where S and T differ
    diff_indices = [i for i in range(n) if s[i] != t[i]]
    
    # M is the number of characters that need to be changed
    m = len(diff_indices)
    print(m)
    
    current_s = list(s)
    
    # To make the array of strings X lexicographically smallest,
    # we need to determine the order in which we change characters.
    # For a specific index i:
    # If T[i] < S[i], changing S[i] to T[i] makes the string lexicographically smaller.
    # If T[i] > S[i], changing S[i] to T[i] makes the string lexicographically larger.
    
    # Strategy for lexicographical smallest array X:
    # 1. First, process all indices i where T[i] < S[i]. 
    #    Among these, process them from left to right (index 0 to n-1) 
    #    because changing a character at a smaller index has a bigger impact on 
    #    making the string lexicographically smaller.
    # 2. Then, process all indices i where T[i] > S[i].
    #    Among these, process them from right to left (index n-1 to 0)
    #    because changing a character at a larger index keeps the string 
    #    lexicographically smaller for longer.
    
    decrease = []
    increase = []
    
    for i in diff_indices:
        if t[i] < s[i]:
            decrease.append(i)
        else:
            increase.append(i)
            
    # Sort decrease indices in ascending order
    decrease.sort()
    # Sort increase indices in descending order
    increase.sort(reverse=True)
    
    # Combine the order of changes
    order = decrease + increase
    
    for idx in order:
        current_s[idx] = t[idx]
        print("".join(current_s))

if __name__ == "__main__":
    solve()