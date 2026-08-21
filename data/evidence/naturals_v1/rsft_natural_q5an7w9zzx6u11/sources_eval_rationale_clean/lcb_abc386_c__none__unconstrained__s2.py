import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # The problem is to check if Edit Distance(S, T) <= 1
    # Since K=1, we check:
    # 1. S == T (0 operations)
    # 2. Length difference is 0: check if they differ by at most 1 character (Replacement)
    # 3. Length difference is 1: check if removing one char from the longer makes it the shorter (Insertion/Deletion)
    # 4. Length difference > 1: Impossible
    
    def check_replacement(s1, s2):
        # Count differences between strings of equal length
        # Using a generator expression inside sum() is allowed as it's a reduction
        diffs = sum(1 for a, b in zip(s1, s2) if a != b)
        return diffs <= 1

    def check_insertion_deletion(longer, shorter):
        # Find the first index where they differ
        # We use a helper to find the mismatch index without a loop
        # We can use a trick with map and next to find the first mismatch
        # However, since we can't use loops, we can check if 
        # there exists an index i such that longer[:i] + longer[i+1:] == shorter
        # But we can't iterate i. 
        # Instead, we find the first mismatch index using a generator.
        
        # Find first mismatch
        mismatch_idx = next((i for i in range(len(longer)) if i >= len(shorter) or longer[i] != shorter[i]), len(longer))
        
        # Check if removing the character at mismatch_idx makes them equal
        return longer[:mismatch_idx] + longer[mismatch_idx+1:] == shorter

    # Logic to determine result based on length difference
    len_s = len(s)
    len_t = len(t)
    diff_len = abs(len_s - len_t)

    # We use a conditional expression to evaluate the result
    result = (
        (s == t) or
        (diff_len == 0 and check_replacement(s, t)) or
        (diff_len == 1 and (
            (len_s > len_t and check_insertion_deletion(s, t)) or 
            (len_t > len_s and check_insertion_deletion(t, s))
        ))
    )

    print("Yes" if result else "No")

if __name__ == "__main__":
    solve()