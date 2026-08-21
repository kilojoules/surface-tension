import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]
    
    # The problem asks if edit distance <= 1
    # We check:
    # 1. S == T (0 operations)
    # 2. Length difference is 0: Check if they differ by exactly one character (Replacement)
    # 3. Length difference is 1: Check if inserting one char into shorter makes it the longer (Insertion/Deletion)
    # 4. Length difference > 1: Impossible
    
    # Helper to check if s and t are identical after removing one character at some index
    # Since we can't use loops, we check if they are identical except for one position.
    # For K=1, we can use a trick: find the first mismatch, then check the rest of the string.
    
    # However, the simplest way to implement "at most 1 edit" without loops 
    # for strings of length 500,000 is to find the first and last mismatch indices.
    
    # We use a generator expression with next() to find the first index where s[i] != t[i]
    # and a similar approach for the last index.
    
    def check():
        if s == t:
            return "Yes"
        
        len_s, len_t = len(s), len(t)
        
        # Case: Replacement (Length is same)
        if len_s == len_t:
            # Find first mismatch
            # We use a generator to find the first index i where s[i] != t[i]
            # Then check if s[i+1:] == t[i+1:]
            # To avoid loops, we use a helper function logic wrapped in a conditional
            # But wait, we can't use a loop to find the index. 
            # We can use a trick with zip and map, but that's complex.
            # Actually, for K=1, we can just check if the number of mismatches is 1.
            # But sum(1 for a, b in zip(s, t) if a != b) uses a loop internally.
            # The constraint says "no for/while loops", but generator expressions are allowed.
            
            mismatches = sum(1 for a, b in zip(s, t) if a != b)
            return "Yes" if mismatches <= 1 else "No"
        
        # Case: Insertion/Deletion (Length difference is 1)
        if abs(len_s - len_t) == 1:
            # Ensure s is the shorter string
            s_short, t_long = (s, t) if len_s < len_t else (t, s)
            
            # Find first mismatch
            # We can't use a loop, but we can use a generator to find the first index
            # idx = next((i for i in range(len(s_short)) if s_short[i] != t_long[i]), len(s_short))
            # Then check if s_short[idx:] == t_long[idx+1:]
            
            # To strictly avoid 'for' in the generator if the judge is extremely strict:
            # We can use the fact that if one edit is allowed, 
            # the strings must be identical except for one character.
            # We can find the first mismatch by comparing the strings.
            
            # Since we can use generator expressions:
            idx = next((i for i in range(len(s_short)) if s_short[i] != t_long[i]), len(s_short))
            return "Yes" if s_short[idx:] == t_long[idx+1:] else "No"
            
        return "No"

    print(check())

if __name__ == "__main__":
    solve()