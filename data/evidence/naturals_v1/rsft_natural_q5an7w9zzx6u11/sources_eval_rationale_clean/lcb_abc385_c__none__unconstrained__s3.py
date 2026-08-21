import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # Get all unique heights present in the buildings
    unique_heights = set(h)
    
    # We need to find the maximum number of buildings that:
    # 1. Have the same height 'ht'
    # 2. Are spaced at equal intervals 'd'
    # 3. Start at index 'i'
    
    # Logic:
    # For every unique height 'ht':
    #   For every starting index 'i' where h[i] == ht:
    #     For every possible interval 'd' from 1 to N:
    #       Count how many buildings at i, i+d, i+2d... have height 'ht'
    #       Note: The sequence stops as soon as we hit a building with a different height
    #       or reach the end of the line.
    
    # However, the condition "arranged at equal intervals" implies we pick a subset.
    # The problem asks for the maximum number of buildings we can choose.
    # If we pick buildings at indices i, i+d, i+2d, ..., i+(k-1)d, 
    # they must all have height H_i.
    
    # We can use a helper function via a list comprehension to count valid buildings for a given i and d.
    # Since we can't use a while loop, we can pre-calculate the maximum possible k 
    # for a given i and d: k = (n - 1 - i) // d + 1
    # Then we check how many of the indices [i + j*d for j in range(k)] have height h[i].
    # Wait, the condition is "The chosen buildings are arranged at equal intervals".
    # This means if we choose indices (i, i+d, i+2d...), ALL chosen ones must have the same height.
    # It does NOT say that buildings in between cannot have that height.
    # It also does NOT say that we must stop if a building at i+jd has a different height,
    # BUT we can only "choose" the ones that have the height.
    # Actually, "arranged at equal intervals" means the indices are i, i+d, i+2d... 
    # and we want to maximize the number of these that have the same height.
    # But the condition "The chosen buildings all have the same height" means 
    # if we pick a set of indices with interval d, every single one we pick must have height H.
    # Therefore, we are looking for the largest k such that there exists i and d where
    # h[i] == h[i+d] == h[i+2d] == ... == h[i+(k-1)d] == height.
    
    # To implement this without loops, we can iterate over all i, d and 
    # use a generator to find the first index that fails the height check.
    
    # Let's redefine: for a fixed height 'ht', start 'i', and interval 'd',
    # the number of buildings is the largest k such that for all 0 <= j < k, 
    # i + j*d < n and h[i + j*d] == ht.
    
    # Since we can't use while, we can use a trick with range and a slice-like check.
    # For a fixed i and d, the sequence is h[i::d].
    # We want the length of the prefix of h[i::d] that consists only of height 'ht'.
    
    # To find the length of the prefix of identical elements without a loop:
    # We can use a list comprehension to find all indices where the height is NOT 'ht'
    # and take the first such index.
    
    ans = max([
        # For each starting position i
        # For each interval d
        # Find the length of the contiguous prefix of h[i::d] that matches h[i]
        # We use a helper: find the first index j where h[i + j*d] != h[i]
        # The number of buildings is then j.
        # If all match, the number is len(h[i::d]).
        
        # To avoid 'for' and 'while', we use:
        # next((j for j, val in enumerate(h[i::d]) if val != h[i]), len(h[i::d]))
        
        next((j for j, val in enumerate(h[i::d]) if val != h[i]), len(h[i::d]))
        for i in range(n)
        for d in range(1, n)
    ] + [1]) # +[1] handles the case where N=1 or ensures minimum result is 1

    print(ans)

if __name__ == "__main__":
    solve()