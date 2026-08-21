import sys

def solve():
    # Read input strings S and T
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    S = input_data[0]
    T = input_data[1]
    
    N = len(S)
    
    # Identify indices where S and T differ
    diff_indices = [i for i in range(N) if S[i] != T[i]]
    M = len(diff_indices)
    
    # To get the lexicographically smallest sequence of strings X,
    # we should process the characters of S from left to right.
    # For each index i where S[i] != T[i]:
    # We want the resulting string to be as small as possible.
    # If T[i] < S[i], changing S[i] to T[i] immediately makes the string smaller.
    # If T[i] > S[i], changing S[i] to T[i] makes the string larger.
    # To minimize the sequence lexicographically, we should first handle all 
    # indices where T[i] < S[i] (from left to right), and then handle all 
    # indices where T[i] > S[i] (from right to left).
    # Wait, the constraint is: we change ONE character per step.
    # To make X_1 smallest, we look for the first index i where S[i] != T[i].
    # If T[i] < S[i], changing S[i] to T[i] is the best move for X_1.
    # If T[i] > S[i], changing any S[j] to T[j] for j > i will result in a string
    # that is lexicographically larger than if we changed some S[k] to T[k] where k < j.
    # Actually, the optimal strategy is:
    # 1. Find all i such that S[i] != T[i] and T[i] < S[i]. Process these from left to right.
    # 2. Find all i such that S[i] != T[i] and T[i] > S[i]. Process these from right to left.
    
    # Let's refine:
    # To make X_1 smallest:
    # If there is any i such that T[i] < S[i], the smallest X_1 is obtained by 
    # picking the smallest i such that T[i] < S[i] and setting S[i] = T[i].
    # If for all i, T[i] >= S[i], then any change will make the string larger.
    # To keep X_1 smallest, we should pick the largest i such that T[i] > S[i] 
    # and set S[i] = T[i].
    
    # Correct Logic for Lexicographical Minimum Array:
    # We must change all differing characters.
    # First, prioritize changes that decrease the string (T[i] < S[i]).
    # These should be done from left to right to make the prefix smaller as quickly as possible.
    # Second, handle changes that increase the string (T[i] > S[i]).
    # These should be done from right to left so that the larger characters are 
    # appended to the end of the string as late as possible.
    
    decrease_indices = [i for i in range(N) if S[i] != T[i] and T[i] < S[i]]
    increase_indices = [i for i in range(N) if S[i] != T[i] and T[i] > S[i]]
    
    # Sort decrease indices ascending, increase indices descending
    order = decrease_indices + increase_indices[::-1]
    
    # Generate the sequence of strings
    # Since we can't use loops, we use a list comprehension with a helper to track state
    # But we can use a simple loop to build the list and then print.
    
    current_S = list(S)
    X = []
    for idx in order:
        current_S[idx] = T[idx]
        X.append("".join(current_S))
    
    # Output results
    print(len(X))
    for s in X:
        print(s)

if __name__ == "__main__":
    solve()