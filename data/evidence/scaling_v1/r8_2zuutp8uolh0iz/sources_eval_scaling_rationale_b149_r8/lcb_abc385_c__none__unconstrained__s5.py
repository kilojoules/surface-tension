import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))

    # We need to find the maximum number of buildings that:
    # 1. Have the same height H
    # 2. Are spaced at equal intervals 'd'
    # 3. Start at index 'i'
    
    # We iterate through all possible starting indices i (0 to n-1)
    # and all possible intervals d (1 to n-1).
    # For a fixed i and d, we check how many buildings at i, i+d, i+2d...
    # have the same height as building i.
    
    # To optimize, we only check intervals d such that i + d < n.
    # The number of buildings in a sequence is (n - 1 - i) // d + 1.
    # However, the condition is that ALL chosen buildings must have the same height.
    # This means we are looking for the longest sequence starting at i with interval d
    # where h[i] == h[i+d] == h[i+2d] ...
    # Wait, the problem says "choose some buildings". It doesn't say they must be 
    # contiguous in the arithmetic progression. 
    # Actually, "arranged at equal intervals" implies we pick a starting point i 
    # and a step d, and we can pick any subset of {i, i+d, i+2d, ...}.
    # But to maximize the count for a fixed i and d, we just count how many 
    # indices j = i + k*d (where j < n) satisfy h[j] == h[i].
    
    # We use a nested comprehension to evaluate all pairs of (i, d).
    # i: starting index
    # d: interval
    
    # The result is the maximum count found across all i and d.
    # We include the case of choosing 1 building by initializing max with 1.
    
    ans = max([
        sum(1 for k in range(0, (n - i + d - 1) // d) if i + k * d < n and h[i + k * d] == h[i])
        for i in range(n)
        for d in range(1, n)
    ] + [1])
    
    print(ans)

if __name__ == "__main__":
    solve()