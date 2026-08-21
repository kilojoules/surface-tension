import sys

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    s = input_data[0]
    t = input_data[1]

    # Identify indices where S and T differ
    # We use a list comprehension to find all i such that s[i] != t[i]
    diff_indices = [i for i, (char_s, char_t) in enumerate(zip(s, t)) if char_s != char_t]
    
    # M is the number of differing characters
    m = len(diff_indices)
    
    # To make the array X lexicographically smallest, we must change the 
    # characters of S to T from left to right.
    # We generate the sequence of strings by applying the changes one by one.
    # We use a list comprehension combined with a slice-based string reconstruction.
    # For the k-th step (0-indexed), the character at diff_indices[k] is updated.
    # The string at step k is S with all characters at diff_indices[0...k] updated to T.
    
    # We create a helper function to build the string for the k-th step
    def get_string_at_step(k):
        # Current indices to be updated are diff_indices[0] through diff_indices[k]
        updated_indices = set(diff_indices[:k+1])
        return "".join([t[i] if i in updated_indices else s[i] for i in range(len(s))])

    # Generate the array X using a list comprehension
    x = [get_string_at_step(k) for k in range(m)]
    
    # Output the results
    sys.stdout.write(f"{m}\n")
    if x:
        sys.stdout.write("\n".join(x) + "\n")

if __name__ == "__main__":
    solve()