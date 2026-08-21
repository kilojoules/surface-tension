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
    
    # To minimize the number of elements in X, we must change exactly one 
    # character per step. The minimum M is the number of differing characters.
    # To make the array X lexicographically smallest, we want the strings 
    # appearing earlier in the array to be lexicographically smaller.
    # This means we should prioritize changing characters at the earliest 
    # possible indices to the target characters in T, BUT only if the 
    # target character T[i] is smaller than the current character S[i].
    # Actually, the rule is simpler: to make the resulting string smallest,
    # we should process indices i where T[i] < S[i] first (from left to right),
    # and then process indices i where T[i] > S[i] (from right to left).
    # Wait, that's for a different problem. Let's re-evaluate.
    
    # For the array X to be lexicographically smallest:
    # We want X_1 to be as small as possible, then X_2, and so on.
    # To make X_1 small, we should change the first index i where S[i] != T[i]
    # IF T[i] < S[i]. If T[i] > S[i], changing it makes the string larger.
    # However, we MUST change one character per step.
    # To keep the string lexicographically smallest, we should:
    # 1. Change indices i where T[i] < S[i] in increasing order of i.
    # 2. Change indices i where T[i] > S[i] in decreasing order of i.
    
    # Let's refine:
    # We want to reach T from S. In each step, we change S[i] to T[i].
    # To make the sequence of strings X lexicographically smallest:
    # - If we change an index i where T[i] < S[i], the string becomes smaller.
    #   We should do this as early as possible, and for the smallest i first.
    # - If we change an index i where T[i] > S[i], the string becomes larger.
    #   We should delay this as much as possible, and for the largest i first.
    
    decrease_indices = [i for i in diff_indices if T[i] < S[i]]
    increase_indices = [i for i in diff_indices if T[i] > S[i]]
    
    # Sort decrease indices ascending, increase indices descending
    order = sorted(decrease_indices) + sorted(increase_indices, reverse=True)
    
    # Generate the sequence of strings
    s_list = list(S)
    x = []
    for i in order:
        s_list[i] = T[i]
        x.append("".join(s_list))
    
    # Output the result
    print(len(x))
    for string in x:
        print(string)

if __name__ == "__main__":
    solve()