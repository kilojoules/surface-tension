import sys

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
    # 1. We want the strings in X to be as small as possible.
    # 2. Changing S[i] to T[i] where S[i] > T[i] makes the string smaller.
    #    We should do these first, and from left to right.
    # 3. Changing S[i] to T[i] where S[i] < T[i] makes the string larger.
    #    We should do these last, and from right to left.
    
    decreasing = [i for i in diff_indices if s_str[i] > t_str[i]]
    increasing = [i for i in diff_indices if s_str[i] < t_str[i]]
    
    # Order of operations: decreasing indices (asc), then increasing indices (desc)
    order = sorted(decreasing) + sorted(increasing, reverse=True)
    
    # We need to generate the sequence of strings. 
    # Since we cannot use loops, we use a recursive-like approach via a list comprehension
    # or a reduction. However, each string depends on the previous one.
    # We can use a helper function with a list to simulate the state.
    
    s_list = list(s_str)
    
    def apply_changes(idx_list):
        # Using a list comprehension to trigger the side effect of updating s_list
        # and returning the current state of the string.
        return [ "".join(s_list).replace(s_list[i], t_str[i], 1) # This is tricky because of duplicates
                 for i in idx_list ]

    # Because the above comprehension doesn't mutate s_list for the next element,
    # we use a function that updates the list and returns the string.
    def get_string(i):
        s_list[i] = t_str[i]
        return "".join(s_list)

    # Use map to apply the function to the order list
    result_strings = list(map(get_string, order))
    
    # Output the number of elements and the strings
    print(len(result_strings))
    print("\n".join(result_strings))

if __name__ == "__main__":
    solve()