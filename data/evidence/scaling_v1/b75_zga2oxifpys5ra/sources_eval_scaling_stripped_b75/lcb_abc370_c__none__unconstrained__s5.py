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
    
    # To get the lexicographically smallest array X:
    # 1. We must change characters one by one (minimum M is the number of differences).
    # 2. At each step, we want the resulting string to be as small as possible.
    # 3. If the target character T[i] is smaller than the current character S[i],
    #    we should prioritize changing that index as early as possible.
    # 4. If the target character T[i] is larger than the current character S[i],
    #    we should delay that change as long as possible.
    
    # Divide indices into those that decrease the character and those that increase it.
    decreasing = [i for i in diff_indices if t[i] < s[i]]
    increasing = [i for i in diff_indices if t[i] > s[i]]
    
    # Sort decreasing indices in increasing order of index to make the string 
    # lexicographically smallest as early as possible.
    # Sort increasing indices in decreasing order of index to delay the 
    # lexicographical increase as much as possible.
    decreasing.sort()
    increasing.sort(reverse=True)
    
    # The sequence of indices to change
    order = decreasing + increasing
    
    # Generate the sequence of strings
    # We use a list comprehension to create the strings by updating the character at the specific index
    # Since we need the state of S after each change, we can't use a simple map.
    # We simulate the process.
    
    def get_sequence(s_list, order_indices, t_list):
        current_s = list(s_list)
        res = []
        for idx in order_indices:
            current_s[idx] = t_list[idx]
            res.append("".join(current_s))
        return res

    x = get_sequence(s, order, t)
    
    # Output the number of elements and the elements themselves
    print(len(x))
    for string in x:
        print(string)

if __name__ == "__main__":
    solve()