import sys

def solve():
    # Read input and handle potential empty lines or extra whitespace
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    K = int(input_data[0])
    S = input_data[1]
    T = input_data[2]
    
    # The problem is to check if the edit distance between S and T is <= K.
    # Since K=1, we have a few specific cases:
    # 1. S == T: distance 0
    # 2. Length difference > 1: distance > 1
    # 3. Length difference == 0: Check if they differ by at most 1 character (Substitution)
    # 4. Length difference == 1: Check if one can be transformed to other by 1 insertion/deletion
    
    if S == T:
        print("Yes")
        return

    len_s = len(S)
    len_t = len(T)

    if abs(len_s - len_t) > 1:
        print("No")
        return

    if len_s == len_t:
        # Check for substitution: count indices where characters differ
        diffs = [i for i in range(len_s) if S[i] != T[i]]
        if len(diffs) <= 1:
            print("Yes")
        else:
            print("No")
    else:
        # Check for insertion/deletion
        # Ensure S is the shorter string
        shorter, longer = (S, T) if len_s < len_t else (T, S)
        
        # Find the first index where they differ
        i = 0
        while i < len(shorter) and shorter[i] == longer[i]:
            i += 1
        
        # If we skip the differing character in the longer string, 
        # the rest of the strings must be identical.
        if shorter[i:] == longer[i+1:]:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    solve()