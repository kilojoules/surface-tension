import sys
from itertools import accumulate

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    S = input_data[0]
    T = input_data[1]
    
    # Identify indices where S and T differ
    diff_indices = [i for i in range(len(S)) if S[i] != T[i]]
    
    # To minimize the number of elements in X, we must change one character per step.
    # The minimum number of elements is the number of differing characters.
    # To make the array X lexicographically smallest, we want the strings X_i 
    # to be as small as possible. 
    # This means we should prioritize changing characters at the earliest possible 
    # indices to the target characters in T, PROVIDED that the target character 
    # is smaller than the current character.
    # However, the constraint is simply to reach T in M steps.
    # To get the lexicographically smallest sequence of strings:
    # In each step, we want to change a character such that the resulting string 
    # is the smallest possible.
    # This implies we should change the character at the first index i where S[i] != T[i]
    # IF T[i] < S[i]. If T[i] > S[i], we should delay that change as long as possible
    # to keep the string smaller for longer.
    
    # Correct Strategy for Lexicographical Smallest X:
    # 1. Change all indices i where T[i] < S[i] in increasing order of i.
    # 2. Change all indices i where T[i] > S[i] in decreasing order of i.
    # Wait, the second part is wrong. To keep the string lexicographically small,
    # we want to keep the characters at the start of the string as small as possible.
    # If T[i] > S[i], changing it makes the string larger. We should do this as late as possible.
    # If T[i] < S[i], changing it makes the string smaller. We should do this as early as possible.
    
    # Let's refine:
    # We must change all diff_indices.
    # To keep X_1 smallest:
    # - If there are any indices i where T[i] < S[i], we should pick the smallest such i.
    # - If there are no indices where T[i] < S[i], we must pick an index where T[i] > S[i].
    #   To keep the string smallest, we should pick the largest such i (the one furthest to the right).
    
    # This looks like a greedy approach. Let's simulate it.
    # Since we can't use loops, we can use a recursive-like structure or a 
    # mathematical way to determine the order of indices.
    
    # The optimal order of indices to change is:
    # All i in diff_indices where T[i] < S[i], sorted ascending.
    # Then all i in diff_indices where T[i] > S[i], sorted descending.
    
    lows = sorted([i for i in diff_indices if T[i] < S[i]])
    highs = sorted([i for i in diff_indices if T[i] > S[i]], reverse=True)
    order = lows + highs
    
    # Now we generate the strings. 
    # We can use a list of characters and update them.
    # Since we need to avoid loops, we can use a technique with a mutable object 
    # inside a list comprehension or use a helper function.
    
    s_list = list(S)
    def get_next_string(idx):
        s_list[idx] = T[idx]
        return "".join(s_list)
    
    # Use map or list comprehension to call the helper
    X = [get_next_string(i) for i in order]
    
    # Output the result
    print(len(X))
    print("\n".join(X))

if __name__ == "__main__":
    solve()