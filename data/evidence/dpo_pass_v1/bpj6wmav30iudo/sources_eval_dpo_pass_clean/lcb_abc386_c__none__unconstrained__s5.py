import sys
from itertools import takewhile

def solve():
    # Read input from stdin
    # input_data[0] is K, input_data[1] is S, input_data[2] is T
    input_data = sys.stdin.read().split()
    
    # Since K=1 is guaranteed by the problem statement
    K = int(input_data[0])
    S = input_data[1]
    T = input_data[2]

    # Find the length of the common prefix
    # takewhile returns an iterator of characters that match
    # we convert to a list to find the length
    common_prefix_len = len(list(takewhile(lambda x: x[0] == x[1], zip(S, T))))

    # Suffixes after the common prefix
    s_suffix = S[common_prefix_len:]
    t_suffix = T[common_prefix_len:]

    # Check if S can be transformed to T in 0 or 1 operations:
    # 0 operations: S equals T
    # 1 operation:
    #   - Replace: len(s_suffix) == 1 and len(t_suffix) == 1
    #   - Delete: len(s_suffix) == 1 and len(t_suffix) == 0
    #   - Insert: len(s_suffix) == 0 and len(t_suffix) == 1
    #   - Shift/Mismatch: If we skip one char in S, does the rest match T?
    #   - Shift/Mismatch: If we skip one char in T, does the rest match S?
    
    # We evaluate the possibilities using a boolean expression
    # Case 0: Identical
    is_identical = (S == T)
    
    # Case 1: Replace (one char differs, lengths are same)
    # S: [prefix] A [suffix], T: [prefix] B [suffix]
    # s_suffix is "A...", t_suffix is "B..."
    # If we replace A with B, then s_suffix[1:] must equal t_suffix[1:]
    is_replace = (len(s_suffix) == len(t_suffix) and 
                  len(s_suffix) > 0 and 
                  s_suffix[1:] == t_suffix[1:])
    
    # Case 2: Delete from S (S is longer by 1)
    # S: [prefix] A [suffix], T: [prefix] [suffix]
    # s_suffix[1:] must equal t_suffix
    is_delete = (len(s_suffix) == len(t_suffix) + 1 and 
                 s_suffix[1:] == t_suffix)
    
    # Case 3: Insert into S (T is longer by 1)
    # S: [prefix] [suffix], T: [prefix] A [suffix]
    # s_suffix must equal t_suffix[1:]
    is_insert = (len(t_suffix) == len(s_suffix) + 1 and 
                 s_suffix == t_suffix[1:])

    # Final result
    result = "Yes" if (is_identical or is_replace or is_delete or is_insert) else "No"
    sys.stdout.write(result + "\n")

if __name__ == "__main__":
    solve()