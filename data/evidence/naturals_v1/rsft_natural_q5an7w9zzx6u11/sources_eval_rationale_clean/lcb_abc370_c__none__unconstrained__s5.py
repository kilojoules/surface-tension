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
    
    # To get the lexicographically smallest array X:
    # 1. We want the strings in X to be as small as possible.
    # 2. Changing S[i] to T[i] where T[i] < S[i] makes the string smaller.
    #    These should be done first, from left to right.
    # 3. Changing S[i] to T[i] where T[i] > S[i] makes the string larger.
    #    These should be done last, from right to left, to keep the 
    #    prefix smaller for as many steps as possible.
    
    decreasing = [i for i in diff_indices if T[i] < S[i]]
    increasing = [i for i in diff_indices if T[i] > S[i]]
    
    # Order of indices to change: 
    # Decreasing (left to right) then Increasing (right to left)
    change_order = decreasing + increasing[::-1]
    
    # We need to generate the sequence of strings.
    # Since we cannot use loops, we use a list comprehension to build the 
    # strings by tracking which indices have been changed.
    # For the k-th string in X, indices change_order[0...k-1] are updated to T.
    
    X = [
        "".join([T[i] if i in change_order[:k] else S[i] for i in range(len(S))])
        for k in range(1, len(change_order) + 1)
    ]
    
    # Output the number of elements and the elements themselves
    sys.stdout.write(f"{len(X)}\n")
    sys.stdout.write("\n".join(X) + ("\n" if X else ""))

if __name__ == "__main__":
    solve()