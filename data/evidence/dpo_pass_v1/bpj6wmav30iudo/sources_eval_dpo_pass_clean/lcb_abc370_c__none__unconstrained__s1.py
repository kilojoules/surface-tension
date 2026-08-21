import sys
from functools import reduce

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    S = input_data[0]
    T = input_data[1]

    # Identify indices where S and T differ
    # We want the lexicographically smallest X.
    # To make X_1 smallest, we should change the first possible character 
    # to the smallest possible value that helps us reach T.
    # However, we can only change ONE character per step.
    # To make the string X_i lexicographically smallest, we should 
    # prioritize changing characters at the earliest possible indices 
    # to their target values in T.
    # But wait: if S[i] > T[i], changing S[i] to T[i] makes the string smaller.
    # If S[i] < T[i], changing S[i] to T[i] makes the string larger.
    # To get the lexicographically smallest array X, we should first 
    # handle all indices i where S[i] > T[i] from left to right, 
    # then handle all indices i where S[i] < T[i] from right to left.
    # Actually, the simplest way to ensure X is lexicographically smallest:
    # 1. Any change that makes the string smaller should be done as early as possible.
    # 2. Any change that makes the string larger should be done as late as possible.
    # Specifically: 
    # - For indices i where S[i] > T[i], we change them to T[i] from left to right.
    # - For indices i where S[i] < T[i], we change them to T[i] from right to left.
    
    diff_indices = [i for i in range(len(S)) if S[i] != T[i]]
    
    # Indices that decrease the string (S[i] > T[i])
    decrease = [i for i in diff_indices if S[i] > T[i]]
    # Indices that increase the string (S[i] < T[i])
    increase = [i for i in diff_indices if S[i] < T[i]]
    
    # Order of operations: 
    # First, all 'decrease' indices from left to right.
    # Then, all 'increase' indices from right to left.
    op_order = decrease + increase[::-1]
    
    # Use reduce to generate the sequence of strings.
    # The accumulator is a list of strings X.
    # The current string S is tracked by taking the last element of the list.
    def apply_op(acc, idx):
        current_s = acc[-1] if acc else S
        # Create new string by replacing character at idx with T[idx]
        new_s = "".join([current_s[i] if i != idx else T[i] for i in range(len(S))])
        return acc + [new_s]

    X = reduce(apply_op, op_order, [])
    
    # Output the result
    sys.stdout.write(f"{len(X)}\n")
    sys.stdout.write("\n".join(X) + ("\n" if X else ""))

if __name__ == "__main__":
    solve()