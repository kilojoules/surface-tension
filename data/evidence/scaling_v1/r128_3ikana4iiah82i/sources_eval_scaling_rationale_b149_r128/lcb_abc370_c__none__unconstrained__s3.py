import sys

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    s_str = input_data[0]
    t_str = input_data[1]
    
    # Identify indices where S and T differ
    # To get the lexicographically smallest array X:
    # 1. We must change characters that are currently larger than the target (S[i] > T[i])
    #    as early as possible to decrease the string's value.
    # 2. We must change characters that are currently smaller than the target (S[i] < T[i])
    #    as late as possible to keep the string's value small for longer.
    
    diff_indices = [i for i in range(len(s_str)) if s_str[i] != t_str[i]]
    
    # Indices to change immediately (where S[i] > T[i])
    # Indices to change later (where S[i] < T[i])
    # We sort the 'decrease' indices in increasing order of index to affect the 
    # most significant positions first.
    # We sort the 'increase' indices in decreasing order of index to affect the
    # least significant positions first.
    
    decrease_indices = sorted([i for i in diff_indices if s_str[i] > t_str[i]])
    increase_indices = sorted([i for i in diff_indices if s_str[i] < t_str[i]], reverse=True)
    
    # The sequence of indices to change is all decreases then all increases
    change_order = decrease_indices + increase_indices
    
    # Use a list comprehension to generate the sequence of strings
    # We use a helper function or a trick to maintain state since we can't use loops.
    # However, since we need to produce a sequence of strings where each depends on the previous,
    # and we cannot use loops or recursion ( own functions), we can use a technique
    # with a mutable list and a list comprehension.
    
    current_s = list(s_str)
    
    # This list comprehension executes the changes and collects the resulting strings.
    # The expression [current_s.__setitem__(i, t_str[i]) or "".join(current_s) for i in change_order]
    # works because __setitem__ returns None, and (None or string) returns the string.
    x = [current_s.__setitem__(i, t_str[i]) or "".join(current_s) for i in change_order]
    
    # Output the results
    print(len(x))
    if x:
        print("\n".join(x))

if __name__ == "__main__":
    solve()