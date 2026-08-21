import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # Identify all unique heights present in the buildings
    unique_heights = set(h)
    
    # For a fixed height 'ht' and a fixed interval 'd', 
    # we want to find the maximum number of buildings of height 'ht' 
    # that appear at indices i, i+d, i+2d...
    # However, the problem says "chosen buildings are arranged at equal intervals".
    # This means we pick a starting index 'i' and a step 'd'.
    # The number of buildings is (n - 1 - i) // d + 1.
    # We need to check if all buildings at those indices have height 'ht'.
    
    # To avoid loops, we use nested list comprehensions.
    # We iterate over all possible heights, all possible starting positions, 
    # and all possible intervals.
    
    # Optimization: Instead of iterating over all heights, we can just 
    # iterate over all pairs (i, j) as the first two buildings of the sequence.
    # The interval is d = j - i. The height is h[i].
    # We then check how many subsequent buildings at interval d also have height h[i].
    
    # Since we cannot use while loops to count, we can use a list comprehension
    # to check the validity of the sequence and then find the length.
    # But wait, the condition is simply that the chosen buildings must have the same height.
    # It doesn't say EVERY building at that interval must be chosen, 
    # but that the ones we DO choose must be at equal intervals.
    # Actually, "arranged at equal intervals" implies if we pick indices p1, p2, ..., pk,
    # then p2 - p1 = p3 - p2 = ... = pk - pk-1 = d.
    
    # For a fixed height 'ht', starting index 'i', and interval 'd':
    # The indices are i, i+d, i+2d... 
    # We count how many of these have height 'ht'. 
    # Wait, the condition is "The chosen buildings all have the same height" 
    # AND "arranged at equal intervals".
    # This means we pick a set of indices {i, i+d, i+2d, ..., i+(k-1)d} 
    # such that H[i] = H[i+d] = ... = H[i+(k-1)d] = ht.
    
    # For a fixed height 'ht', we find all indices where H[idx] == ht.
    # Then for every pair of these indices (i, j), we check the interval d = j - i.
    # We check how many points in the arithmetic progression starting at i with diff d 
    # also have height 'ht'.
    
    # Let's refine:
    # For every height 'ht' in the set:
    #   Indices L = [idx for idx, val in enumerate(h) if val == ht]
    #   For every pair i, j in L (i < j):
    #     d = j - i
    #     Count k such that i + (k-1)d < n and h[i + (k-1)d] == ht
    
    # To implement "count k" without a loop:
    # We can use a list comprehension to check all possible k and sum the booleans.
    # Max k is n // d + 1.
    
    # However, the simplest way to think about it:
    # For every starting index i and every interval d (1 <= d < n):
    #   Height ht = h[i]
    #   Count how many k >= 0 satisfy i + k*d < n and h[i + k*d] == ht.
    
    # Using list comprehensions to find the maximum:
    ans = max([
        sum([1 for k in range((n - 1 - i) // d + 1) if h[i + k * d] == h[i]])
        for i in range(n)
        for d in range(1, n)
    ] + [1]) # +[1] handles the case where N=1 or only 1 building is chosen
    
    print(ans)

if __name__ == "__main__":
    solve()