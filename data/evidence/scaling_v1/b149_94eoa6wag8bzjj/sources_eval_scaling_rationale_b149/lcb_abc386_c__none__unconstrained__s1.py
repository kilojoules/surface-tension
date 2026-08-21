import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # The problem is to check if edit distance between S and T is <= 1.
    # Since K=1 is guaranteed, we check:
    # 1. S == T (0 operations)
    # 2. Length difference is 0: Check if they differ by exactly one character (Replacement)
    # 3. Length difference is 1: Check if removing one char from the longer string makes it the shorter one (Insertion/Deletion)
    # 4. Length difference > 1: Impossible
    
    # Helper to check if two strings are identical after removing one character at some index
    # For K=1, we can use a generator expression to check all possible single-character removals.
    # However, a more efficient way for K=1 is to find the first mismatch and check the remaining suffixes.
    
    def check_one_diff(s1, s2):
        # Checks if s1 and s2 are identical or differ by exactly one replacement
        return sum(1 for a, b in zip(s1, s2) if a != b) <= 1

    def check_insertion(shorter, longer):
        # Checks if inserting one character into 'shorter' can produce 'longer'
        # We find the first mismatch and check if the rest of the strings match
        # Since we cannot use loops, we use a recursive-like logic via slicing 
        # but the constraint allows generator expressions.
        # For K=1, we can simply check if there's an index i where 
        # shorter[:i] + shorter[i:] == longer[:i] + longer[i+1:]
        # But that's O(N^2). Instead, we find the first mismatch.
        
        # To avoid loops and recursion, we find the first index of mismatch
        # using a generator and next().
        mismatches = [i for i in range(len(shorter)) if shorter[i] != longer[i]]
        # This is still a loop. The constraint says "no for/while loops".
        # Let's use a different approach.
        pass

    # Correct approach without loops:
    # 1. If S == T, Yes.
    # 2. If len(S) == len(T), check if they differ by at most 1 char.
    # 3. If abs(len(S) - len(T)) == 1, check if the shorter is a subsequence of the longer.
    #    For strings of length N and N-1, being a subsequence is equivalent to edit distance 1.
    
    # To check subsequence without loops:
    # We can use a recursive-like structure or built-ins. 
    # Actually, the most reliable way to check if S is a subsequence of T 
    # when len(T) = len(S) + 1 is to find the first index where they differ,
    # then check if S[i:] == T[i+1:].
    
    # Since I cannot use loops, I will use a helper function with a 
    # conditional expression that handles the logic.
    
    def is_subseq(s1, s2):
        # s1 is shorter, s2 is longer.
        # Find first mismatch index
        # We can't use a loop to find the index, but we can use 
        # a generator expression inside next() to find the first index.
        try:
            idx = next(i for i in range(len(s1)) if s1[i] != s2[i])
            return s1[idx:] == s2[idx+1:]
        except StopIteration:
            return True

    # Final logic wrapped in a print statement
    print("Yes" if (
        (s == t) or
        (len(s) == len(t) and sum(1 for a, b in zip(s, t) if a != b) <= 1) or
        (len(s) == len(t) + 1 and is_subseq(t, s)) or
        (len(t) == len(s) + 1 and is_subseq(s, t))
    ) else "No")

if __name__ == "__main__":
    solve()