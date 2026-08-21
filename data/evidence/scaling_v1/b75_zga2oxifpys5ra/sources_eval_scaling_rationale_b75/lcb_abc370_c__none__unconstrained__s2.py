import sys
from itertools import accumulate

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    s_str = input_data[0]
    t_str = input_data[1]
    
    # Identify indices where S and T differ
    diff_indices = [i for i in range(len(s_str)) if s_str[i] != t_str[i]]
    m = len(diff_indices)
    
    # To get the lexicographically smallest array X:
    # We must change characters at indices i such that the resulting string is minimized.
    # If S[i] > T[i], changing S[i] to T[i] makes the string lexicographically smaller.
    # If S[i] < T[i], changing S[i] to T[i] makes the string lexicographically larger.
    # Therefore, we should process indices where S[i] > T[i] first (from left to right),
    # then indices where S[i] < T[i] (from right to left).
    
    decreasing = [i for i in diff_indices if s_str[i] > t_str[i]]
    increasing = [i for i in diff_indices if s_str[i] < t_str[i]]
    
    # Order of indices to change:
    # 1. Indices where S[i] > T[i], processed from left to right.
    # 2. Indices where S[i] < T[i], processed from right to left.
    order = decreasing + increasing[::-1]
    
    # We use accumulate to generate the sequence of strings.
    # The function takes the current string and the index to change, 
    # returning a new string with the character at that index replaced by T[index].
    def transform(current_s, idx):
        return current_s[:idx] + t_str[idx] + current_s[idx+1:]

    # accumulate(iterable, func) applies the functionK times.
    # We pass the order of indices and use the initial S as the starting point.
    # Since accumulate yields the result of each step, we get the sequence X.
    x = list(accumulate(order, transform, initial=s_str))
    
    # The first element of accumulate(..., initial=s_str) is the initial S.
    # The problem asks for the array X containing the strings AFTER each change.
    result = x[1:]
    
    # Output the number of elements and each string in X.
    # Using join and map to avoid for-loops for printing.
    sys.stdout.write(f"{m}\n" + "\n".join(result) + ("\n" if m > 0 else ""))

if __name__ == "__main__":
    solve()