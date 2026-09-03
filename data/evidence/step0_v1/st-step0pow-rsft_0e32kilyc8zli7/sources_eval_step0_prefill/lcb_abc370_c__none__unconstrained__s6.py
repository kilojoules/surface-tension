import sys

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    s = list(input_data[0])
    t = list(input_data[1])
    n = len(s)

    # Identify indices where S and T differ
    diff_indices = [i for i in range(n) if s[i] != t[i]]
    m = len(diff_indices)

    # To minimize the number of elements in X, we must change exactly one 
    # character per step. The minimum number of steps is the number of 
    # positions where S and T differ.
    # To make the array X lexicographically smallest, we want the strings 
    # X_1, X_2, ... X_M to be as small as possible.
    # This means we should prioritize changing characters at indices that 
    # make the resulting string lexicographically smaller first.
    # Specifically, if T[i] < S[i], changing S[i] to T[i] makes the string 
    # smaller. We should do these changes as early as possible, from left to right.
    # If T[i] > S[i], changing S[i] to T[i] makes the string larger. 
    # We should delay these changes as much as possible, from right to left.

    # Indices that make the string smaller (T[i] < S[i])
    decrease = [i for i in diff_indices if t[i] < s[i]]
    # Indices that make the string larger (T[i] > S[i])
    increase = [i for i in diff_indices if t[i] > s[i]]

    # Sort decrease indices ascending (left to right) to get the smallest string fastest
    decrease.sort()
    # Sort increase indices descending (right to left) to delay the increase of the string
    increase.sort(reverse=True)

    # The sequence of indices to change
    change_order = decrease + increase

    # Construct the array X
    x = []
    current_s = s[:]
    for idx in change_order:
        current_s[idx] = t[idx]
        x.append("".join(current_s))

    # Output the result
    print(len(x))
    for string in x:
        print(string)

if __name__ == "__main__":
    solve()