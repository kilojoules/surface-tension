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
    
    # The edit distance between S and T must be <= K.
    # For K=1, the possible scenarios are:
    # 1. S == T (0 operations)
    # 2. Length difference is 0: One character differs (1 replacement)
    # 3. Length difference is 1: One character inserted or deleted (1 operation)
    # 4. Length difference > 1: Impossible
    
    # Case 0: Identical
    if s == t:
        print("Yes")
        return

    # Case 1: Replacement (Length must be equal)
    if n == m:
        # Count positions where characters differ
        diffs = sum(1 for a, b in zip(s, t) if a != b)
        if diffs <= k:
            print("Yes")
        else:
            print("No")
        return

    # Case 2: Insertion/Deletion (Length difference must be exactly 1)
    if abs(n - m) == 1:
        # Identify which string is shorter and which is longer
        shorter = s if n < m else t
        longer = t if n < m else s
        
        # We need to check if 'shorter' is a subsequence of 'longer' 
        # with only one character difference.
        # Since K=1 and length diff is 1, we just need to check if 
        # removing one char from 'longer' makes it 'shorter'.
        
        # Find the first index where they differ
        # Use a generator to find the first mismatch
        diff_idx = next((i for i in range(len(shorter)) if shorter[i] != longer[i]), len(shorter))
        
        # If we skip the character at diff_idx in the longer string, 
        # the rest must match the shorter string.
        if shorter[diff_idx:] == longer[diff_idx + 1:]:
            print("Yes")
        else:
            print("No")
        return

    # Case 3: Length difference > 1
    print("No")

if __name__ == "__main__":
    solve()