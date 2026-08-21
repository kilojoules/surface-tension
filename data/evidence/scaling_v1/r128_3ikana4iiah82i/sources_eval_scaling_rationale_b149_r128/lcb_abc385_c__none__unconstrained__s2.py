import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))

    # We need to find the maximum number of buildings that:
    # 1. Have the same height.
    # 2. Are spaced at equal intervals.
    
    # Let i be the starting index (0 to n-1)
    # Let d be the interval (1 to n-1)
    # For a fixed i and d, we check how many buildings at indices i, i+d, i+2d... 
    # have the same height as building i.
    
    # We use a generator expression inside max() to evaluate all combinations.
    # The inner sum() counts how many buildings in the sequence match the height of the first building.
    # range(i, n, d) generates the indices at equal intervals.
    
    # Note: If n=1, the loops for d will not execute, so we handle the base case.
    # However, the logic naturally handles n=1 if we ensure the result is at least 1.
    
    # We iterate through all possible starting points i and all possible intervals d.
    # For each pair (i, d), we calculate the length of the sequence of buildings 
    # that share the same height as building i.
    
    # To avoid explicit loops, we use nested comprehensions.
    # We calculate the count for every i and d, then take the maximum.
    
    # Since we must check if the buildings are the SAME height, 
    # we check h[j] == h[i] for j in range(i, n, d).
    # BUT, the condition says "the chosen buildings all have the same height".
    # This means if we pick an interval d, we can only pick buildings that match 
    # the height of the first one we picked. 
    # If we encounter a building with a different height, we cannot simply "skip" it;
    # the equal interval condition applies to the indices of the chosen buildings.
    # Therefore, for a fixed i and d, we are checking the sequence h[i], h[i+d], h[i+2d]...
    # and we want to know the maximum number of these that have the same height.
    # Actually, the problem implies we choose a subset of buildings. 
    # If we choose buildings at indices i, i+d, i+2d, then ALL of them must have the same height.
    # So for a fixed i and d, we count how many k exist such that h[i + k*d] == h[i].
    # Wait, the condition "arranged at equal intervals" means the indices are i, i+d, i+2d...
    # It does NOT say we can skip some. It means the set of indices is {i + k*d | 0 <= k < m}.
    # For this set to be valid, h[i] == h[i+d] == h[i+2d] ... == h[i+(m-1)*d].
    
    # Correct logic: For every starting index i and every interval d,
    # find the largest m such that h[i] == h[i+d] == ... == h[i+(m-1)*d].
    
    # To implement this without loops, we can use a helper function or 
    # a clever comprehension. Since we can't use while loops, we can 
    # pre-calculate the lengths for all i, d.
    
    # For a fixed i and d, the number of buildings is:
    # count = 1
    # while i + count*d < n and h[i + count*d] == h[i]:
    #     count += 1
    
    # Without while loops, we can use a list comprehension to create a boolean 
    # mask for the sequence and then find the length of the prefix of True values.
    # However, a simpler way: for a fixed i and d, the maximum m is the 
    # number of elements in the sequence h[i], h[i+d]... that are equal to h[i]
    # BEFORE the first element that is NOT equal to h[i] appears.
    
    # Let's use a different approach:
    # For every i and d, we generate the sequence [h[i], h[i+d], h[i+2d], ...]
    # We want the length of the prefix where all elements == h[i].
    
    # Since N is small (3000), we can't do O(N^3). O(N^2) is required.
    # The number of pairs (i, d) is N^2. 
    # For each pair, we need to find the prefix length.
    
    # Actually, the most efficient way to solve this is:
    # For every pair of indices (i, j) with i < j:
    # If h[i] == h[j], they could be the first two buildings of a sequence.
    # The interval is d = j - i.
    # But we need to know if this is an extension of a previous sequence.
    # This looks like DP: dp[i][d] = length of sequence ending at i with interval d.
    # dp[i][d] = dp[i-d][d] + 1 if h[i] == h[i-d] else 1.
    
    # To implement DP without loops, we can use a dictionary or a list 
    # and process indices in order. But we can't use for loops.
    # We can use `functools.reduce`.
    
    from functools import reduce
    
    # dp state: a dictionary where key is (index, interval) and value is length.
    # We iterate through indices 0 to n-1.
    
    def update_dp(acc, i):
        # For the current index i, we check all possible intervals d.
        # d can range from 1 to i.
        # We update the DP table for (i, d).
        # Since we can't use a loop, we use a dictionary comprehension.
        
        # Current heights: h[i]
        # For each d in 1...i:
        # if h[i] == h[i-d]: dp[i, d] = dp.get((i-d, d), 0) + 1
        # else: dp[i, d] = 1
        
        # To avoid creating a massive dictionary, we only store (i, d) if h[i] == h[i-d].
        # Otherwise, the length is just 1.
        
        # We use a dictionary comprehension to find all d such that h[i] == h[i-d].
        updates = {d: acc.get((i-d, d), 0) + 1 
                   for d in range(1, i + 1) 
                   if h[i] == h[i-d]}
        
        # Merge updates into acc. 
        # We only need to store the ones that are > 1.
        return {**acc, **{(i, d): val for d, val in updates.items()}}

    # This is still O(N^2). Let's refine it.
    # The result is the max value in the DP table, or 1 if the table is empty.
    
    # Using reduce to simulate the loop over i:
    final_dp = reduce(update_dp, range(n), {})
    
    if not final_dp:
        print(1)
    else:
        print(max(final_dp.values()))

if __name__ == "__main__":
    solve()