import sys

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    S = input_data[0]
    T = input_data[1]
    
    # Identify indices where S and T differ
    diff_indices = [i for i in range(len(S)) if S[i] != T[i]]
    
    # To make the array X lexicographically smallest:
    # 1. We want the first string X1 to be as small as possible.
    # 2. If we can change a character S[i] to T[i] such that T[i] < S[i],
    #    doing this at the smallest index i will make the string smallest.
    # 3. If we must change S[i] to T[i] where T[i] > S[i], we want to delay
    #    this as much as possible, or do it at the rightmost index to keep
    #    the prefix small.
    
    # Indices where S[i] > T[i] should be processed first (from left to right)
    # Indices where S[i] < T[i] should be processed last (from right to left)
    decreasing = [i for i in diff_indices if S[i] > T[i]]
    increasing = [i for i in diff_indices if S[i] < T[i]]
    
    # Sort decreasing indices ascending (left to right)
    # Sort increasing indices descending (right to left)
    order = sorted(decreasing) + sorted(increasing, reverse=True)
    
    # Use a list comprehension to generate the sequence of strings
    # We use a helper function with a mutable list to simulate state 
    # since we cannot use loops or recursion (per constraints)
    # Actually, we can use a reducer-like approach with a list 
    # and a side-effect inside a list comprehension.
    
    current_s = list(S)
    
    def get_next_string(idx):
        current_s[idx] = T[idx]
        return "".join(current_s)
    
    # Generate the array X by applying the changes in the determined order
    X = [get_next_string(i) for i in order]
    
    # Output the result
    print(len(X))
    print("\n".join(X))

if __name__ == "__main__":
    solve()