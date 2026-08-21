import sys

def solve():
    # Read input from standard input
    try:
        input_data = sys.stdin.read().split()
        if not input_data:
            return
        
        k = int(input_data[0])
        s = input_data[1]
        t = input_data[2]
    except (EOFError, IndexError):
        return

    # The problem asks if the edit distance (Levenshtein distance) 
    # between S and T is <= K.
    # For K=1, we can check this efficiently.
    
    # If the difference in lengths is more than 1, edit distance > 1
    if abs(len(s) - len(t)) > 1:
        print("No")
        return

    # If they are already identical
    if s == t:
        print("Yes")
        return

    # Case 1: Lengths are equal (Possible operation: Replace)
    if len(s) == len(t):
        diff_count = 0
        for i in range(len(s)):
            if s[i] != t[i]:
                diff_count += 1
            if diff_count > 1:
                print("No")
                return
        print("Yes" if diff_count <= 1 else "No")
        return

    # Case 2: Lengths differ by 1 (Possible operation: Insert or Delete)
    # Let 'shorter' be the string with length n, 'longer' be length n+1
    if len(s) < len(t):
        shorter, longer = s, t
    else:
        shorter, longer = t, s

    # Check if shorter is a subsequence of longer by removing exactly one char
    # We use a two-pointer approach to see if shorter can be formed from longer
    i = 0 # pointer for shorter
    j = 0 # pointer for longer
    diffs = 0
    
    while i < len(shorter) and j < len(longer):
        if shorter[i] == longer[j]:
            i += 1
            j += 1
        else:
            diffs += 1
            j += 1 # "skip" the character in the longer string
            if diffs > 1:
                print("No")
                return
    
    # If we reached the end of shorter, it means we could match it 
    # by removing at most one character from longer.
    print("Yes")

if __name__ == "__main__":
    solve()