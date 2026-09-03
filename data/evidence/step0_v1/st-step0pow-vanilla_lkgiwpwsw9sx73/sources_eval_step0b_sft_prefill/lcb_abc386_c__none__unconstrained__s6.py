import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]
    
    n, m = len(s), len(t)
    
    # The edit distance between two strings S and T is the minimum number of operations
    # to transform S into T. We need to check if edit_distance(S, T) <= K.
    # Since K=1, we can check this without a full DP table.
    
    # Case 0: S and T are already identical
    if s == t:
        print("Yes")
        return

    # Case 1: Length difference is more than 1
    if abs(n - m) > 1:
        print("No")
        return

    # Case 2: Length difference is 1 (Insertion or Deletion)
    if abs(n - m) == 1:
        # Ensure s is the shorter string
        shorter, longer = (s, t) if n < m else (t, s)
        
        # Find the first index where they differ
        # We use a trick with slicing or a loop to find the mismatch
        # Since we can't use loops, we use a list comprehension to find the first mismatch
        mismatches = [i for i in range(len(shorter)) if shorter[i] != longer[i]]
        
        if not mismatches:
            # Shorter is a prefix of longer
            print("Yes")
        else:
            idx = mismatches[0]
            # Check if removing the character at idx from 'longer' makes it 'shorter'
            # longer[:idx] + longer[idx+1:] == shorter
            if longer[:idx] + longer[idx+1:] == shorter:
                print("Yes")
            else:
                print("No")
        return

    # Case 3: Lengths are equal (Replacement or identical)
    if n == m:
        # Count mismatches
        mismatches = [i for i in range(n) if s[i] != t[i]]
        if len(mismatches) <= 1:
            print("Yes")
        else:
            print("No")
        return

if __name__ == "__main__":
    solve()