import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # If strings are already identical, it's 0 operations
    if s == t:
        print("Yes")
        return

    # Since K=1, we only need to check if the edit distance is exactly 1.
    # We check three cases:
    # 1. Replacement: lengths are equal, only one character differs.
    # 2. Deletion from S: len(s) == len(t) + 1, removing one char from s makes it t.
    # 3. Insertion into S: len(s) + 1 == len(t), removing one char from t makes it s.
    
    # Helper to check if s can become t by removing one character at some index i
    # We use a generator expression inside any() to avoid explicit loops.
    def check_one_removal(longer, shorter):
        # We find the first mismatch
        # To avoid loops, we can't easily find the index 'i' without a loop,
        # but we can use a trick: find the first index where they differ.
        # However, since we can't use loops, we can use a generator to check all split points.
        # But wait, checking all split points with a generator is allowed.
        return any(longer[:i] + longer[i+1:] == shorter for i in range(len(longer)))

    # For replacement, we check if they differ by exactly one character.
    def check_replacement(s1, s2):
        if len(s1) != len(s2):
            return False
        # Count mismatches using a generator expression
        return sum(1 for a, b in zip(s1, s2) if a != b) == 1

    # Logic for K=1
    # We use a conditional expression to determine the result.
    result = "Yes" if (
        (len(s) == len(t) and check_replacement(s, t)) or
        (len(s) == len(t) + 1 and check_one_removal(s, t)) or
        (len(t) == len(s) + 1 and check_one_removal(t, s))
    ) else "No"

    print(result)

if __name__ == "__main__":
    solve()