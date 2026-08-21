import sys

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    S = input_data[0]
    T = input_data[1]
    
    # Identify indices where S and T differ
    # To get the lexicographically smallest sequence of strings,
    # we must decide the order in which we change characters.
    # If S[i] < T[i], changing S[i] to T[i] makes the string lexicographically larger.
    # If S[i] > T[i], changing S[i] to T[i] makes the string lexicographically smaller.
    # To keep the sequence smallest, we should:
    # 1. Process indices where S[i] > T[i] first, from left to right.
    # 2. Process indices where S[i] < T[i] last, from right to left.
    
    diff_indices = [i for i in range(len(S)) if S[i] != T[i]]
    
    # Indices where S[i] > T[i] (decreasing the character)
    # These should be handled first, from left to right, to make the string smaller as quickly as possible.
    dec_indices = [i for i in diff_indices if S[i] > T[i]]
    
    # Indices where S[i] < T[i] (increasing the character)
    # These should be handled last, from right to left, to keep the prefix smaller for longer.
    inc_indices = [i for i in diff_indices if S[i] < T[i]]
    inc_indices.sort(reverse=True)
    
    order = dec_indices + inc_indices
    
    # Recursive function to generate the sequence of strings
    def generate_sequence(current_s, indices):
        if not indices:
            return []
        
        idx = indices[0]
        # Create new string by replacing character at idx with T[idx]
        # Using join and list comprehension to avoid loops
        new_s = "".join([current_s[i] if i != idx else T[i] for i in range(len(current_s))])
        
        return [new_s] + generate_sequence(new_s, indices[1:])

    result = generate_sequence(S, order)
    
    # Output the number of steps and the sequence
    print(len(result))
    print("\n".join(result))

if __name__ == "__main__":
    solve()