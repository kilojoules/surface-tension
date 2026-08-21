import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # K is given as 1 for this sub-problem
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]
    
    # If the strings are already identical, 0 operations are needed.
    if s == t:
        print("Yes")
        return

    # Since K=1, we only need to check if the edit distance is 1.
    # Edit distance is 1 if:
    # 1. One character is replaced (lengths are equal)
    # 2. One character is deleted (len(S) = len(T) + 1)
    # 3. One character is inserted (len(S) = len(T) - 1)

    len_s = len(s)
    len_t = len(t)

    # Case 1: Replace (Length must be the same)
    if len_s == len_t:
        diff_count = 0
        for i in range(len_s):
            if s[i] != t[i]:
                diff_count += 1
            if diff_count > 1:
                break
        if diff_count == 1:
            print("Yes")
            return

    # Case 2: Delete from S / Insert into T (len_s = len_t + 1)
    elif len_s == len_t + 1:
        # Try to find one character to remove from s to make it t
        i = 0
        while i < len_t and s[i] == t[i]:
            i += 1
        # After the first mismatch, the rest of s[i+1:] must match t[i:]
        if s[i+1:] == t[i:]:
            print("Yes")
            return

    # Case 3: Insert into S / Delete from T (len_s = len_t - 1)
    elif len_s == len_t - 1:
        # Try to find one character to remove from t to make it s
        i = 0
        while i < len_s and s[i] == t[i]:
            i += 1
        # After the first mismatch, the rest of s[i:] must match t[i+1:]
        if s[i:] == t[i+1:]:
            print("Yes")
            return

    # If none of the conditions are met, it's impossible with 1 operation
    print("No")

if __name__ == "__main__":
    solve()