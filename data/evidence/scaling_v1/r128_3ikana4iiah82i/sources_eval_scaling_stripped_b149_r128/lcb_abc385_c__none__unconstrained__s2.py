import sys
from itertools import groupby

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # Group indices by building height
    # height_groups = {height: [index1, index2, ...]}
    # We use a dictionary comprehension to organize indices of the same height
    # To avoid loops, we can use a technique with groupby or a comprehension
    # Since we can't use loops, we create a list of (height, index) pairs and sort them
    indexed_heights = sorted([(h[i], i) for i in range(n)])
    
    # Group by height and extract only the indices
    # groups will be a list of lists: [[indices_of_height1], [indices_of_height2], ...]
    groups = [list(map(lambda x: x[1], g)) for k, g in groupby(indexed_heights, lambda x: x[0])]
    
    # For each group of indices, we want to find the maximum number of indices 
    # that form an arithmetic progression.
    # An arithmetic progression is defined by a starting index 'a' and a common difference 'd'.
    # The number of elements is the count of indices i in the group such that 
    # (i - a) is divisible by d and the quotient is within the range of the group's size.
    # However, the condition is "equal intervals" in the original line, 
    # meaning the indices themselves must form an arithmetic progression.
    
    # To find the max length of an AP in each group without loops:
    # For a fixed group of indices 'idx_list', and a fixed difference 'd',
    # we can count how many elements belong to the same congruence class modulo 'd'.
    # But the elements must be present in the list.
    
    # Let's refine: for every pair of indices (i, j) in a group, they define a difference d = j - i.
    # We want to find how many k in the group satisfy k = i + m*d.
    
    # Since N is 3000, an O(N^2) approach is acceptable.
    # We can iterate through all possible differences d from 1 to N.
    # For a fixed d, we can check all starting positions.
    
    # Actually, the simplest way to implement this without explicit 'for' loops 
    # is using comprehensions.
    
    # For each group, we check all possible differences d (from 1 to N)
    # and all possible starting indices in that group.
    # But that's O(N^3). We need O(N^2).
    
    # Correct O(N^2) approach:
    # For each group, iterate through all pairs (i, j) to define d = j - i.
    # Then count how many elements in the group fit the pattern.
    # Wait, that's still O(N^3) if we count in a loop.
    
    # Let's use the property: for a fixed d, we can group indices by (index % d).
    # For a specific height, the max number of buildings at interval d is:
    # max(count of indices in group that are congruent mod d AND form a contiguous range)
    # Actually, the condition "equal intervals" means indices are i, i+d, i+2d...
    # This means we are looking for the longest sequence of indices in the group 
    # that form an arithmetic progression.
    
    # Let's use a different approach:
    # For each height group, we can use a dictionary to store the length of the AP 
    # ending at index i with difference d.
    # dp[i][d] = dp[i-d][d] + 1 if height[i] == height[i-d] else 1.
    # Since we can't use loops, we can use a reduction or a clever comprehension.
    
    # However, the simplest O(N^2) is:
    # For each possible difference d in [1, N]:
    #   For each height h in [1, 3000]:
    #     Find the longest run of indices in the group of height h that are separated by d.
    
    # Let's use the DP approach with a dictionary and a reduction.
    # We process indices from 0 to N-1.
    # state = { (index, diff): length }
    # This is still tricky without loops.
    
    # Let's go back to: for each group, for each pair (i, j), d = j - i.
    # The number of elements is (j - i) // d + 1 IF all intermediate elements exist.
    # But we can just check if (i + k*d) is in the set for k=0, 1, ...
    
    # Given the constraints and the "no loop" rule, the most reliable way to 
    # implement the DP is using a dictionary and `functools.reduce`.
    from functools import reduce
    
    # We want to compute dp[i][d] = dp[i-d][d] + 1
    # We can represent the DP table as a dictionary where keys are (i, d)
    # The reduce function will iterate through i from 0 to N-1.
    
    def update_dp(acc, i):
        # For the current index i, we check all possible differences d.
        # d can range from 1 to i.
        # If h[i] == h[i-d], then dp[i][d] = dp[i-d][d] + 1.
        # Otherwise, dp[i][d] = 1.
        # We only need to store entries in acc that are > 1 to save space.
        
        # We create a dictionary of updates for the current i
        current_updates = {
            d: acc.get((i - d, d), 0) + 1 
            for d in range(1, i + 1) 
            if h[i] == h[i - d]
        }
        
        # Merge current_updates into acc
        # Since we can't use a loop to update acc, we create a new dictionary
        # But we only need to store the results for the current i.
        # Actually, we can just return the updated dictionary.
        return {**acc, **{(i, d): val for d, val in current_updates.items()}}

    # To avoid the dictionary growing too large and to keep it O(N^2),
    # we can use a list of dictionaries or a similar structure.
    # But the most straightforward way is to use a list comprehension 
    # inside the reduce to build the new state.
    
    # Let's refine the reduce:
    # The state is a dictionary: (index, diff) -> length
    # For each i, we calculate lengths for all d and add them to the dictionary.
    
    final_dp = reduce(update_dp, range(n), {})
    
    # The answer is the maximum value in the DP table.
    # If the table is empty, the answer is 1.
    print(max(final_dp.values()) if final_dp else 1)

# Standard Python entry point
if __name__ == "__main__":
    solve()