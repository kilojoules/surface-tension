import sys
from functools import reduce

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    s = input_data[0]
    t = input_data[1]
    
    # Identify indices where S and T differ
    # We use a list comprehension to find all i where s[i] != t[i]
    diff_indices = [i for i in range(len(s)) if s[i] != t[i]]
    
    # The number of operations M is the number of differing characters
    m = len(diff_indices)
    
    # To generate the sequence of strings X, we need to change characters one by one.
    # We can use reduce to maintain the current state of the string and collect the history.
    # reduce(function, sequence, initial)
    # The accumulator will be a list of strings [current_s, X1, X2, ...]
    
    def evolve(acc, idx):
        current_s = acc[-1]
        # Create a new string by changing the character at idx to t[idx]
        # Using slicing to avoid loops
        next_s = current_s[:idx] + t[idx] + current_s[idx+1:]
        acc.append(next_s)
        return acc

    # Start reduce with the initial string s in the list
    # We map the evolve function over the diff_indices
    result_list = reduce(evolve, diff_indices, [s])
    
    # The first element of result_list is the initial S, which is not part of X.
    # X consists of the strings generated after each change.
    x = result_list[1:]
    
    # Output the result
    sys.stdout.write(f"{m}\n")
    if x:
        sys.stdout.write("\n".join(x) + "\n")

if __name__ == "__main__":
    solve()