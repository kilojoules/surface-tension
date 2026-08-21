import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # We need to find the maximum number of buildings with the same height 
    # that are spaced at equal intervals.
    # Let i be the starting building, j be the second building.
    # The interval is (j - i). We check how many buildings at 
    # i, i + (j-i), i + 2(j-i)... have the same height H_i.
    
    # We use a list comprehension to iterate through all possible 
    # starting positions i and all possible intervals d.
    # For a fixed i and d, we count how many k satisfy h[i + k*d] == h[i].
    # However, the condition is that ALL chosen buildings must have the same height.
    # If we encounter a building with a different height, the sequence breaks.
    
    # To avoid loops, we can use a generator expression inside max().
    # For every pair (i, j) where i < j, if h[i] == h[j], we check the sequence.
    # Since N is small (3000), a O(N^2) approach is acceptable.
    # Actually, checking every i and every interval d is O(N^2).
    
    # We can use a helper logic: for a fixed i and interval d, 
    # the number of buildings is the length of the prefix of the 
    # sequence (i, i+d, i+2d...) that all have height h[i].
    
    # Since we cannot use while/for loops, we use a trick with 
    # range and a generator to find the first index that fails.
    # But wait, the problem says "the chosen buildings", implying we can 
    # pick a subset. "Arranged at equal intervals" usually means 
    # indices i, i+d, i+2d... 
    # The condition "The chosen buildings all have the same height" 
    # means we are looking for the maximum k such that 
    # h[i] == h[i+d] == h[i+2d] == ... == h[i+(k-1)d].
    
    # We can use a list comprehension to evaluate all i and d.
    # For a fixed i and d, we find the number of elements 
    # until the first height mismatch using a generator and next().
    
    ans = max(
        (
            # For each starting position i and interval d
            # We find the first k where h[i + k*d] != h[i]
            # The number of buildings is k.
            next(
                (k for k in range(1, n // d + 2) if (i + k * d >= n or h[i + k * d] != h[i])),
                n // d + 1
            )
            for i in range(n)
            for d in range(1, n)
            if i + d < n and h[i] == h[i + d]
        ),
        # Default value if no pairs are found
        1
    )
    
    print(ans)

if __name__ == "__main__":
    solve()