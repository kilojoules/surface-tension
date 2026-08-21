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
    # To achieve this, we should change characters at indices that 
    # result in the smallest possible string at each step.
    # Since we must change one character per step, and we want the 
    # resulting string to be lexicographically smallest, we should 
    # prioritize changing characters at the earliest possible positions 
    # if the target character T[i] is smaller than S[i].
    # However, the problem asks for the lexicographically smallest ARRAY X.
    # This means X[0] should be as small as possible, then X[1], etc.
    # To make X[0] smallest, we should change the first character that 
    # differs between S and T to T[i], provided T[i] < S[i].
    # If T[i] > S[i] for the first differing index, we cannot make the 
    # string smaller by changing that index; we should look for any 
    # index i where T[i] < S[i] to make the string smaller.
    # Actually, the simplest strategy to get the lexicographically smallest 
    # string at each step is: 
    # 1. Find all indices i where S[i] != T[i].
    # 2. To make the current S smaller, we want to change the leftmost 
    #    index i where T[i] < S[i].
    # 3. If no such i exists (all T[i] > S[i] for differing indices), 
    #    we must change the rightmost index i where T[i] > S[i] to 
    #    keep the string as small as possible for as long as possible.
    
    # Let's refine: we have a set of indices that MUST be changed.
    # To make the sequence X lexicographically smallest:
    # At each step, we want the resulting string to be the smallest possible.
    # We can change any index i where S[i] != T[i] to T[i].
    # To make the string smallest, we prioritize indices i where T[i] < S[i]
    # from left to right. Once all T[i] < S[i] are handled, we handle 
    # T[i] > S[i] from right to left.
    
    diff_indices = [i for i in range(len(S)) if S[i] != T[i]]
    
    # Indices that make the string smaller (T[i] < S[i]) sorted ascending
    # Indices that make the string larger (T[i] > S[i]) sorted descending
    order = [i for i in diff_indices if T[i] < S[i]] + \
            [i for i in sorted(diff_indices, reverse=True) if T[i] > S[i]]
    
    # Use reduce to generate the sequence of strings.
    # The accumulator is a tuple: (current_string, list_of_X)
    result = reduce(
        lambda acc, idx: (
            # Create new string by replacing char at idx
            # Using a list comprehension to join since we can't use loops
            "".join([acc[0][j] if j != idx else T[j] for j in range(len(S))]),
            # Append the new string to the list
            acc[1] + ["".join([acc[0][j] if j != idx else T[j] for j in range(len(S))])]
        ),
        order,
        (S, [])
    )
    
    X = result[1]
    
    # Output the result
    sys.stdout.write(f"{len(X)}\n" + "\n".join(X) + ("\n" if X else ""))

if __name__ == "__main__":
    solve()