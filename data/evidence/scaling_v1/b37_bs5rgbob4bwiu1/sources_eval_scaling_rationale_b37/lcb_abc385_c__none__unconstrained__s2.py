import sys

def solve():
    # Read N and the list of heights
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # We need to find the maximum number of buildings that:
    # 1. Have the same height H_i
    # 2. Are spaced at equal intervals 'd'
    # 3. Start at index 'i'
    
    # We can iterate over all possible starting indices i (0 to N-1)
    # and all possible intervals d (1 to N-1).
    # For a fixed i and d, we check how many buildings H[i], H[i+d], H[i+2d]...
    # have the same height as H[i].
    # Since the condition is that ALL chosen buildings must have the same height,
    # and they must be at equal intervals, we stop counting as soon as we hit 
    # a building with a different height.
    
    # However, the problem asks for the maximum number of buildings.
    # A simpler way to think about it: for every pair (i, d), 
    # we count how many k >= 0 exist such that i + k*d < N and H[i + k*d] == H[i].
    # Wait, the condition "arranged at equal intervals" implies we pick a 
    # sequence i, i+d, i+2d... i+(m-1)d. 
    # For this sequence to be valid, H[i] = H[i+d] = ... = H[i+(m-1)d].
    
    # To implement this without loops, we can use nested comprehensions.
    # We iterate over all starting positions i and all possible intervals d.
    # For each (i, d), we find the length of the contiguous sequence of 
    # buildings with the same height.
    
    # Since we cannot use loops, we can use a helper function with recursion 
    # or a comprehension that simulates the counting.
    # Actually, the most "functional" way to count the prefix of matches 
    # is to use a list comprehension to identify all indices that match the height,
    # and then find the length of the initial segment that follows the arithmetic progression.
    
    # Correct logic: For a fixed start i and interval d, we are looking for the 
    # largest m such that H[i] == H[i+d] == H[i+2d] == ... == H[i+(m-1)d].
    
    # We can use a recursive function to count the matches for a specific i and d.
    def count_matches(i, d, height):
        if i >= N or H[i] != height:
            return 0
        return 1 + count_matches(i + d, d, height)

    # To avoid recursion depth issues and stick to the "no loop" rule using comprehensions:
    # We can pre-calculate the answers for all i, d.
    # Since N is 3000, a full O(N^2) is acceptable. 
    # We can iterate i from 0 to N-1 and d from 1 to N.
    # For a fixed i and d, the number of buildings is the number of k such that
    # H[i + k*d] == H[i] for all 0 <= k < m.
    
    # Using a list comprehension to evaluate all (i, d) pairs:
    # We use a trick: for a fixed i and d, we can create a list of booleans 
    # [H[i] == H[i+k*d] for k in range(...)].
    # Then we find the index of the first False.
    
    # However, the simplest way to solve this within constraints is to realize 
    # that for a fixed i and d, we want the smallest k such that H[i+kd] != H[i].
    # The answer for (i, d) is that k.
    
    # To avoid explicit loops, we use map/max/comprehensions.
    # We only need to check d from 1 to N.
    # For each i and d, we can use a generator to find the first index where height differs.
    
    # Because we cannot use 'while', we can use a recursive function with @lru_cache
    # or simply use the fact that we can use range() inside comprehensions.
    
    # Let's use a recursive approach to count the sequence length.
    import sys
    sys.setrecursionlimit(5000)
    
    # memoized function to count consecutive matches
    from functools import lru_cache
    @lru_cache(None)
    def get_len(i, d):
        if i >= N:
            return 0
        # We are checking the sequence starting at i with interval d.
        # This function is called for the sequence starting at i.
        # It checks if the current one matches the start of the sequence.
        # But the start height is determined by the original i.
        # So we need the height as a parameter.
        return 0 # Placeholder

    # Revised recursive function
    @lru_cache(None)
    def count_seq(i, d, height):
        if i >= N or H[i] != height:
            return 0
        return 1 + count_seq(i + d, d, height)

    # We want max(count_seq(i, d, H[i])) for all i in 0..N-1, d in 1..N
    # To avoid the overhead of calling the function N^2 times in a comprehension,
    # we can use a comprehension to trigger the calculations.
    
    # Since we need to avoid loops, we use map and max.
    # We iterate i from 0 to N-1 and d from 1 to N.
    # Note: d can go up to N.
    
    ans = max(
        map(
            lambda i: max(
                (count_seq(i, d, H[i]) for d in range(1, N + 1)),
                default=1
            ),
            range(N)
        ),
        default=0
    )
    
    print(ans)

if __name__ == "__main__":
    solve()