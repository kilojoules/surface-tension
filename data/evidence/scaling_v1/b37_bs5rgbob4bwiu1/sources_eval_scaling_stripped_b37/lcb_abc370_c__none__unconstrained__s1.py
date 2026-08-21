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
    
    # To get the lexicographically smallest sequence of strings:
    # 1. We must change characters one by one.
    # 2. For a specific index i, if T[i] < S[i], changing S[i] to T[i] 
    #    makes the string lexicographically smaller immediately.
    #    Therefore, we should process all indices where T[i] < S[i] first,
    #    and process them in increasing order of index.
    # 3. If T[i] > S[i], changing S[i] to T[i] makes the string lexicographically larger.
    #    To keep the sequence smallest, we should process these last,
    #    and process them in decreasing order of index (to keep the prefix smaller for longer).
    
    # Indices where T[i] < S[i] (Increasing order)
    first_batch = [i for i in diff_indices if t[i] < s[i]]
    first_batch.sort()
    
    # Indices where T[i] > S[i] (Decreasing order)
    second_batch = [i for i in diff_indices if t[i] > s[i]]
    second_batch.sort(reverse=True)
    
    # The sequence of indices to change
    change_order = first_batch + second_batch
    
    # Generate the sequence of strings
    # We use a list comprehension to create the strings by updating the state of s
    # Since we cannot use loops, we use a helper function with reduce or a custom map
    # However, the simplest way to maintain state without a loop is using a generator 
    # or a function that updates the list and returns the current join.
    
    def apply_change(current_s, idx):
        current_s[idx] = t[idx]
        return "".join(current_s)

    # Using map to iterate through the change_order and update the list s
    # map() will execute the apply_change function for each index in change_order
    result_x = list(map(lambda i: apply_change(s, i), change_order))
    
    # Output the number of elements
    print(len(result_x))
    # Output each string in the array X
    if result_x:
        print("\n".join(result_x))

if __name__ == "__main__":
    solve()