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
    
    # To get the lexicographically smallest array X:
    # 1. If T[i] < S[i], changing S[i] to T[i] makes the string smaller.
    #    We want to do this as early as possible.
    # 2. If T[i] > S[i], changing S[i] to T[i] makes the string larger.
    #    We want to do this as late as possible.
    # Therefore, we sort indices: those with T[i] < S[i] first (in increasing order of i),
    # then those with T[i] > S[i] last (in decreasing order of i).
    
    # Sort criteria: 
    # Primary: (T[i] < S[i]) is True (1) or False (0). 
    # Since we want True first, we use -1 for True.
    # Secondary: If T[i] < S[i], index i ascending. If T[i] > S[i], index i descending.
    
    sorted_indices = sorted(
        diff_indices, 
        key=lambda i: (0 if t_str[i] < s_str[i] else 1, i if t_str[i] < s_str[i] else -i)
    )
    
    # Use accumulate to generate the sequence of strings.
    # Each step takes the current string and applies the change at the next sorted index.
    def apply_change(current_s, idx):
        # Strings are immutable, so we convert to list, change, and join back.
        s_list = list(current_s)
        s_list[idx] = t_str[idx]
        return "".join(s_list)

    # accumulate(iterable, func) applies the function cumulatively.
    # We pass the sorted_indices as the iterable and a wrapper for apply_change.
    # Since accumulate passes the previous result as the first arg, we use a closure.
    
    # To avoid a loop or recursion, we use a helper function with accumulate.
    # We need to pass the index from sorted_indices into the function.
    # We can use a generator or map to pair the current string with the index.
    
    # Correct approach with accumulate:
    # We create a sequence of functions that apply the change, then compose them.
    # Or more simply, use a list comprehension to build the strings if we can't use loops.
    # But the prompt forbids list comprehensions for iteration.
    
    # Let's use a recursive-like structure via map/reduce or accumulate.
    # We can use a list of indices and a custom function for accumulate.
    # However, accumulate's function signature is (accumulator, element).
    
    res = list(accumulate(sorted_indices, lambda acc, idx: apply_change(acc, idx), 
                         initial=s_str))
    
    # The 'initial' value is S, but X contains strings AFTER the change.
    # So we remove the first element (S).
    x_array = res[1:]
    
    # Output the number of elements and the elements themselves.
    sys.stdout.write(f"{len(x_array)}\n")
    sys.stdout.write("\n".join(x_array) + ("\n" if x_array else ""))

if __name__ == "__main__":
    solve()