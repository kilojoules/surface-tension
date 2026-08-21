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
    diff_indices = [i for i in range(len(s)) if s[i] != t[i]]
    
    # To get the lexicographically smallest sequence of strings:
    # 1. We want the strings in X to be as small as possible.
    # 2. Changing a character to a smaller one makes the string smaller.
    # 3. We should process decreases (S[i] > T[i]) from left to right.
    # 4. We should process increases (S[i] < T[i]) from right to left.
    
    decreases = [i for i in diff_indices if s[i] > t[i]]
    increases = [i for i in diff_indices if s[i] < t[i]]
    
    # Sort decreases ascending, increases descending
    order = sorted(decreases) + sorted(increases, reverse=True)
    
    # Use reduce to simulate the process of changing characters and collecting strings
    # accumulator: (current_string, list_of_strings)
    def apply_change(acc, idx):
        curr_s, x_list = acc
        # Create new string by replacing character at idx with T[idx]
        new_s = curr_s[:idx] + t[idx] + curr_s[idx+1:]
        return (new_s, x_list + [new_s])
    
    final_state = reduce(apply_change, order, (s, []))
    result_x = final_state[1]
    
    # Output the number of elements and the elements themselves
    print(len(result_x))
    # Using join and map to avoid explicit loops for printing
    sys.stdout.write('\n'.join(result_x) + ('\n' if result_x else ''))

if __name__ == "__main__":
    solve()