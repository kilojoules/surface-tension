import sys
from functools import reduce

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    s_str = input_data[0]
    t_str = input_data[1]

    # Identify indices where S and T differ
    diff_indices = [i for i in range(len(s_str)) if s_str[i] != t_str[i]]
    
    # To get the lexicographically smallest array X, we must change characters
    # at indices that make the resulting string smaller as early as possible.
    # 1. If T[i] < S[i], we want to change this index as early as possible.
    # 2. If T[i] > S[i], we want to change this index as late as possible.
    # Therefore, we process indices where T[i] < S[i] in increasing order,
    # followed by indices where T[i] > S[i] in decreasing order.
    
    decreasing_indices = [i for i in diff_indices if t_str[i] < s_str[i]]
    increasing_indices = [i for i in diff_indices if t_str[i] > s_str[i]]
    
    # Sort decreasing_indices ascending (to fix 'larger' chars to 'smaller' ones first)
    # Sort increasing_indices descending (to fix 'smaller' chars to 'larger' ones last)
    order = sorted(decreasing_indices) + sorted(increasing_indices, reverse=True)
    
    # Use reduce to generate the sequence of strings.
    # The accumulator is a list of strings (the array X).
    def evolve(acc, idx):
        current_s = acc[-1] if acc else s_str
        # Create a new string by changing the character at the given index
        next_s = current_s[:idx] + t_str[idx] + current_s[idx+1:]
        return acc + [next_s]

    result_x = reduce(evolve, order, [])
    
    # Output the number of elements and the elements themselves
    print(len(result_x))
    # Use join and map to avoid explicit loops for printing
    if result_x:
        sys.stdout.write('\n'.join(result_x) + '\n')

if __name__ == "__main__":
    solve()