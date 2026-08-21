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
    
    # We iterate through:
    # i: starting index (0 to n-1)
    # d: interval (1 to n-1)
    # For a fixed i and d, we check how many buildings at i, i+d, i+2d... 
    # have the same height as building i.
    
    # The condition "chosen buildings are arranged at equal intervals" 
    # implies we pick indices i, i+d, i+2d, ... i+kd.
    # All these must have height H_i.
    
    # We use a generator expression inside max() to evaluate all possibilities.
    # For a fixed start i and interval d, the number of buildings is:
    # count = length of the sequence [i, i+d, i+2d, ...] such that H_j == H_i.
    # However, the problem says "the chosen buildings" must satisfy the condition.
    # This means we can't just skip a building in the sequence. 
    # If we pick interval d, we check indices i, i+d, i+2d... and stop 
    # the moment H_{i+kd} != H_i.
    
    # Wait, the condition "arranged at equal intervals" means the indices 
    # are an arithmetic progression. It does NOT say we must stop at the 
    # first mismatch; it says the SET of chosen buildings must be at equal intervals.
    # Therefore, for a fixed i and d, we can pick all indices j = i + k*d 
    # such that H_j == H_i.
    
    # Let's refine: 
    # For every pair of indices (i, j) where i < j and H_i == H_j:
    # The interval is d = j - i.
    # We can then check all k such that i + k*d < n and H_{i+kd} == H_i.
    
    # To avoid loops, we use comprehensions.
    # We can iterate over all possible starts i and all possible intervals d.
    
    # result = max(
    #    count of k where (i + k*d < n and h[i + k*d] == h[i])
    #    for i in range(n)
    #    for d in range(1, n)
    # )
    
    # Since we need to count how many k satisfy the condition for a FIXED i and d:
    # We can use a list comprehension to gather all valid k's and take the length.
    
    # Optimization: we only care about d if h[i] == h[i+d].
    # But for N=3000, N^2 is 9 million, which might be slow in Python.
    # However, the constraint to avoid loops forces this approach.
    
    # Let's use a more efficient approach:
    # For each height, find all indices where it occurs.
    # Then for every pair of indices in that set, calculate the interval.
    
    # But the simplest comprehension is:
    # max(
    #   len([k for k in range(0, (n-i-1)//d + 1) if h[i + k*d] == h[i]])
    #   for i in range(n)
    #   for d in range(1, n)
    # )
    # Note: if d is large, the inner list is small. If d=1, it's large.
    # The total complexity is roughly N^2 * (N/d), which is N^2 log N.
    # For N=3000, this might be too slow. 
    # Let's optimize: we only need to check d that are divisors of (j-i).
    
    # Actually, the most direct way to implement the logic:
    # For every i and d, count how many k satisfy h[i + k*d] == h[i].
    # To keep it within time limits, we can iterate i and d, 
    # but we only check d if h[i] == h[i+d].
    
    # Using a generator to find the maximum:
    ans = max(
        [
            len([k for k in range(0, (n - i - 1) // d + 1) if h[i + k * d] == h[i]])
            for i in range(n)
            for d in range(1, n) if i + d < n and h[i] == h[i + d]
        ] + [1] # Handle the case where no two buildings have the same height
    )
    
    print(ans)

if __name__ == "__main__":
    solve()