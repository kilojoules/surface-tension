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
    
    # We can iterate over all possible starting indices 'i' (0 to N-1)
    # and all possible intervals 'd' (1 to N-1).
    # For a fixed i and d, we check how many buildings at indices i, i+d, i+2d...
    # have the same height as H[i].
    
    # To avoid loops, we use nested list comprehensions.
    # For a fixed i and d, the sequence of indices is range(i, N, d).
    # We count how many indices j in that range satisfy H[j] == H[i].
    # However, the condition says "chosen buildings are arranged at equal intervals".
    # This means if we pick indices i, i+d, i+2d..., they must ALL have the same height
    # to be counted together, OR we are looking for the longest subsequence 
    # with a constant gap where all elements are identical.
    
    # Actually, the condition is: we pick a set of indices {i, i+d, i+2d, ... i+(k-1)d}.
    # All these must have the same height.
    # For a fixed i and d, we can check the maximum k such that H[i] == H[i+d] == ... == H[i+(k-1)d].
    
    # Since N is small (3000), a O(N^2) approach is acceptable.
    # We can iterate over all pairs (i, j) as the first two buildings of the sequence.
    # The interval is d = j - i. We then check how many subsequent buildings H[j+d], H[j+2d]...
    # have the same height as H[i].
    
    # Using a helper function to count the length of the sequence for a given i and d:
    # We can use a recursive-like structure via a list comprehension or simply 
    # check all possible k values.
    
    # A more efficient way without loops:
    # For every pair (i, j) where H[i] == H[j], the interval is d = j - i.
    # We want to find the largest k such that H[i + m*d] == H[i] for all 0 <= m < k.
    
    # However, the simplest way to express this in a comprehension is:
    # For every starting point i and every interval d:
    # Find the first m where H[i + m*d] != H[i]. The count is m.
    
    # Since we cannot use while loops, we can use a trick with 
    # next() and a generator expression to find the first index that fails.
    
    ans = max(
        [
            # For a fixed i and d, find the length of the contiguous block of identical heights
            # We use next() to find the first index m where the height differs, 
            # and provide a default value (N // d + 1) if all match.
            next(
                (m for m in range(1, (N - i) // d + 1) if H[i + m * d] != H[i]),
                (N - i - 1) // d + 1
            )
            for i in range(N)
            for d in range(1, N)
            if i + d < N # Only check if a second building is possible
        ] + [1] # Ensure at least 1 is returned
    )
    
    print(ans)

if __name__ == "__main__":
    solve()