import sys

def solve():
    # Read input and handle potential empty lines or extra whitespace
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # The problem asks if S can be transformed to T in 0 to K operations.
    # Since K=1, we check if S == T or if they differ by exactly one edit.
    
    # Case 0: Already identical
    if s == t:
        print("Yes")
        return

    # Case 1: Replacement (Lengths must be equal)
    # Check if they differ by exactly one character
    def check_replace():
        if len(s) != len(t):
            return False
        # Count positions where characters differ
        diffs = sum(1 for a, b in zip(s, t) if a != b)
        return diffs == 1

    # Case 2: Deletion from S / Insertion into S
    # If len(s) == len(t) + 1, we check if removing one char from s makes it t
    # If len(t) == len(s) + 1, we check if removing one char from t makes it s
    def check_single_diff(longer, shorter):
        if len(longer) != len(shorter) + 1:
            return False
        
        # Find the first index where they differ
        # We use a generator to find the first mismatch index
        # Since we can't use loops, we find the mismatch by comparing slices
        # or by finding the first index i where longer[i] != shorter[i]
        
        # To avoid loops, we can use a trick: 
        # If we remove character at index i, the remaining parts must match.
        # However, checking all i would be O(N^2). 
        # Instead, we find the first mismatch and check if skipping it works.
        
        # Finding the first mismatch index without a loop:
        # We can use a generator expression with next()
        mismatch_idx = next((i for i in range(len(shorter)) if longer[i] != shorter[i]), len(shorter))
        
        # Check if removing the character at mismatch_idx makes them equal
        return longer[:mismatch_idx] + longer[mismatch_idx+1:] == shorter

    # Combine all checks
    # We use 'or' to short-circuit the evaluation
    result = (
        (s == t) or 
        check_replace() or 
        (len(s) == len(t) + 1 and check_single_diff(s, t)) or 
        (len(t) == len(s) + 1 and check_single_diff(t, s))
    )

    print("Yes" if result else "No")

if __name__ == "__main__":
    solve()