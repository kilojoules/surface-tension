import sys

def solve():
    # Read N and the sequence A
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    A = input_data[1:]
    
    # last_seen stores the most recent 1-based index of each value encountered
    last_seen = {}
    
    # Helper function to process each element
    # It updates the dictionary and returns the previous index
    def get_prev_index(item_tuple):
        idx, val = item_tuple
        # Get the previous index or -1 if not seen before
        prev = last_seen.get(val, -1)
        # Update the dictionary with the current 1-based index
        last_seen[val] = idx
        return prev

    # Use enumerate to get 1-based indices and map to apply the helper function
    # enumerate(A, 1) provides (1, A_1), (2, A_2), ...
    # map() applies the function to every element of the iterable
    B = map(get_prev_index, enumerate(A, 1))
    
    # Print the results joined by spaces
    print(*(B))

if __name__ == "__main__":
    solve()