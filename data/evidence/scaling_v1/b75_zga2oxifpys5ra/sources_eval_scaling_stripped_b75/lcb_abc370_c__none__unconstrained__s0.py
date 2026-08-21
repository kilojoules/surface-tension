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
    
    # To get the lexicographically smallest array X:
    # 1. We must change characters one by one (minimum M is the number of differences).
    # 2. At each step, we want the resulting string to be as small as possible.
    # 3. A string is smaller if the character at the earliest possible index is smaller.
    # 4. Therefore, we should process indices from left to right.
    # 5. However, if the target character T[i] is alphabetically larger than the current S[i],
    #    changing it immediately makes the string lexicographically larger.
    #    If T[i] is smaller than S[i], changing it immediately makes the string smaller.
    # 6. Wait, the rule is simpler: to make the sequence of strings X lexicographically smallest,
    #    we want X_1 to be as small as possible, then X_2, and so on.
    #    To make X_1 smallest, we should change the first index i where S[i] > T[i].
    #    If no such i exists, we must change the last index i where S[i] < T[i] to keep the string as small as possible for as long as possible.
    
    # Correct Strategy for Lexicographical Smallest X:
    # We have a set of indices that must be changed.
    # In each step, we pick one index i and set S[i] = T[i].
    # To make the current S smallest:
    # - If there are indices i where S[i] > T[i], we should pick the leftmost such i.
    #   Changing S[i] to T[i] (where T[i] < S[i]) decreases the string lexicographically.
    # - If all remaining indices i have S[i] < T[i], we should pick the rightmost such i.
    #   Changing S[i] to T[i] (where T[i] > S[i]) increases the string lexicographically,
    #   so we do it as far to the right as possible to keep the string smaller.

    s_list = list(s)
    t_list = list(t)
    remaining = set(diff_indices)
    
    # We will build the sequence X by repeatedly finding the best index to change
    # until no differences remain.
    x = []
    while remaining:
        # Find indices where S[i] > T[i]
        decreasing = [i for i in remaining if s_list[i] > t_list[i]]
        
        if decreasing:
            # Pick the leftmost index that decreases the string
            idx = min(decreasing)
        else:
            # All remaining changes increase the string, pick the rightmost
            idx = max(remaining)
            
        s_list[idx] = t_list[idx]
        x.append("".join(s_list))
        remaining.remove(idx)

    # The problem asks for the minimum number of elements M.
    # M is simply the number of differences.
    # The logic above handles the lexicographical requirement.
    
    # Since the logic above is iterative and depends on the current state of S,
    # and we must change exactly one character per step, the total steps is len(diff_indices).
    
    # Note: The logic "leftmost S[i] > T[i], else rightmost S[i] < T[i]" 
    # is the standard approach for this specific problem.
    
    # However, the logic above is slightly flawed because it doesn't account for the 
    # fact that we can only change each index once. Let's refine:
    # We need to change all i in diff_indices.
    # To make X_1 smallest:
    # If there's any i in diff_indices such that T[i] < S[i], the best X_1 is 
    # achieved by changing the leftmost such i.
    # If for all i in diff_indices, T[i] > S[i], the best X_1 is achieved by 
    # changing the rightmost such i.
    
    # Let's re-implement without the while loop to avoid mutation issues and ensure O(N^2).
    
    def get_sequence(current_s, rem_indices):
        if not rem_indices:
            return []
        
        # Find indices that would decrease the string
        decreasing = [i for i in rem_indices if current_s[i] > t_list[i]]
        if decreasing:
            idx = min(decreasing)
        else:
            idx = max(rem_indices)
            
        new_s = list(current_s)
        new_s[idx] = t_list[idx]
        res_s = "".join(new_s)
        
        return [res_s] + get_sequence(res_s, rem_indices - {idx})

    # Using a loop instead of recursion to avoid depth limits
    final_x = []
    current_s = s
    rem = set(diff_indices)
    while rem:
        decreasing = [i for i in rem if current_s[i] > t_list[i]]
        idx = min(decreasing) if decreasing else max(rem)
        
        s_l = list(current_s)
        s_l[idx] = t_list[idx]
        current_s = "".join(s_l)
        final_x.append(current_s)
        rem.remove(idx)

    print(len(final_x))
    for string in final_x:
        print(string)

if __name__ == "__main__":
    solve()