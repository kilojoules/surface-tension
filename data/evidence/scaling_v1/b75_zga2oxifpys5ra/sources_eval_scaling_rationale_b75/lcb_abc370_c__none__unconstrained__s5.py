import sys

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    s = input_data[0]
    t = input_data[1]
    
    # Identify indices where S and T differ
    diff_indices = [i for i in range(len(s)) if s[i] != t[i]]
    
    # To get the lexicographically smallest sequence of strings:
    # 1. Characters that decrease (S[i] > T[i]) should be changed as early as possible.
    # 2. Characters that increase (S[i] < T[i]) should be changed as late as possible.
    # 3. Among decreasing characters, those appearing earlier in the string should be changed first.
    # 4. Among increasing characters, those appearing later in the string should be changed first.
    
    decreasing = [i for i in diff_indices if s[i] > t[i]]
    increasing = [i for i in diff_indices if s[i] < t[i]]
    
    # Sort decreasing indices in ascending order (left to right)
    # Sort increasing indices in descending order (right to left)
    order = sorted(decreasing) + sorted(increasing, reverse=True)
    
    # We need to generate the sequence of strings. 
    # Since we cannot use loops or recursion, we use a list comprehension 
    # combined with a helper function or a reduction-like approach.
    # However, since we need the state of the string at each step, 
    # we can pre-calculate which character is changed at each step.
    
    # We use a list comprehension to build the strings.
    # For the k-th string in the sequence, characters at indices in order[:k+1] are replaced by T.
    
    # Function to build the string for a given set of changed indices
    get_string = lambda changed_indices: "".join(
        [t[i] if i in changed_indices else s[i] for i in range(len(s))]
    )
    
    # Generate the sequence of changed index sets
    # We use a list comprehension to create a list of sets of indices changed so far
    # But since we can't use loops to maintain state, we can use a range and slice.
    x = [get_string(set(order[:k+1])) for k in range(len(order))]
    
    # Output the results
    print(len(x))
    print("\n".join(x))

if __name__ == "__main__":
    solve()