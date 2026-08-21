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
    # i: the starting index (0 to n-1)
    # d: the interval (1 to n-1)
    # For a fixed i and d, we check how many buildings at indices i, i+d, i+2d... 
    # have the same height as building i.
    # Since the condition is "all chosen buildings have the same height",
    # and they must be at equal intervals, we only need to check if 
    # h[i] == h[i + k*d] for k = 0, 1, 2...
    # The sequence stops as soon as we hit a building of a different height 
    # or the end of the line.
    
    # However, the problem says "the chosen buildings" must satisfy this.
    # This implies we can pick a subset of buildings at interval d.
    # Wait, "arranged at equal intervals" usually means the gap between 
    # consecutive chosen buildings is constant.
    # If we pick indices i, i+d, i+2d, ..., i+(k-1)d, they are at equal intervals.
    # All these must have the same height.
    
    # To maximize k, for every pair (i, d), we count how many indices 
    # j = i + k*d (where j < n) satisfy h[j] == h[i].
    # IMPORTANT: The condition "arranged at equal intervals" means the 
    # distance between any two adjacent chosen buildings is the same.
    # This means we are looking for the largest k such that there exists 
    # i and d where h[i] == h[i+d] == h[i+2d] == ... == h[i+(k-1)d].
    
    # We can use a helper function to count the length of the valid sequence.
    # Since we can't use while loops, we can use a list comprehension to 
    # find all indices j = i + k*d and then find the first index that fails.
    # But a simpler way: for a fixed i and d, we check all k such that i+k*d < n.
    # The sequence of chosen buildings must be i, i+d, i+2d... 
    # If we encounter a building with a different height, we cannot 
    # skip it and continue because the "equal interval" must be maintained 
    # between the chosen ones. 
    # Actually, the problem says "The chosen buildings are arranged at equal intervals."
    # This means if we choose indices {p1, p2, ..., pk}, then p_{j+1} - p_j = d.
    # So we are looking for the maximum k such that h[i] == h[i+d] == ... == h[i+(k-1)d].
    
    # For a fixed i and d, the number of buildings is the number of 
    # consecutive elements starting from i with step d that have height h[i].
    # Since we can't use while, we can use a trick with itertools.takewhile 
    # or just check all k and find the first failure.
    
    # Let's use a comprehension to evaluate all possible (i, d) pairs.
    # For a fixed i and d, we want to find the largest k such that 
    # for all 0 <= m < k, i + m*d < n and h[i + m*d] == h[i].
    
    # We can use a list comprehension to get the sequence of booleans 
    # [h[i] == h[i + m*d] for m in range((n-i-1)//d + 1)]
    # Then we need the length of the prefix of True values.
    
    # To find the length of the prefix of True values without a loop:
    # We can use a generator and next() or just iterate and find the first False.
    # But we can just iterate through all possible k and check the condition.
    
    # Optimization: Instead of checking prefixes, just iterate through all 
    # i, d and count how many j = i + k*d satisfy h[j] == h[i].
    # WAIT: The condition is "The chosen buildings are arranged at equal intervals."
    # This means if we pick indices {p1, ..., pk}, then p2-p1 = p3-p2 = ... = d.
    # It does NOT say we cannot have a building of a different height at 
    # index i + 0.5d. It only cares about the buildings we ACTUALLY choose.
    # So for a fixed i and d, we just count how many j = i + k*d (for k=0, 1, ...)
    # satisfy h[j] == h[i]. 
    # NO, that's wrong. If we choose indices {0, 2, 4} and h[0]=5, h[2]=5, h[4]=5,
    # they are at equal intervals (d=2) and have the same height. 
    # It doesn't matter if h[1] or h[3] are different.
    # BUT, the chosen buildings must be the ONLY ones we pick.
    # So for a fixed i and d, we can pick all j = i + k*d such that h[j] == h[i].
    # Wait, the condition "arranged at equal intervals" means the distance 
    # between *consecutive* chosen buildings is the same.
    # If we pick indices {0, 4, 8}, the interval is 4. 
    # If we pick {0, 2, 4, 6, 8}, the interval is 2.
    # If we pick {0, 2, 8}, they are NOT at equal intervals.
    # Therefore, for a fixed i and d, we can pick ALL j = i + k*d 
    # as long as h[j] == h[i]. 
    # But if h[i+d] != h[i], we can't just skip it and pick h[i+2d].
    # Because then the interval between the 1st and 2nd chosen building 
    # would be 2d, while the interval between others might be d.
    # Actually, if we pick {i, i+2d, i+4d}, the interval is 2d.
    # So for any pair (i, d), we just need to count how many k >= 0 
    # satisfy i + k*d < n AND h[i + k*d] == h[i].
    # This is NOT correct. If we have h = [5, 7, 5, 7, 5], and we pick 
    # indices 0, 2, 4, the interval is 2. All have height 5. This is valid.
    # If we have h = [5, 7, 7, 7, 5], and we pick indices 0, 4, the interval is 4.
    # This is also valid.
    # So for every pair (i, d), we check the sequence i, i+d, i+2d... 
    # and count how many of them have height h[i].
    # BUT the condition "arranged at equal intervals" means the 
    # distance between *adjacent* chosen buildings is constant.
    # If we pick indices p1 < p2 < ... < pk, then p2-p1 = p3-p2 = ... = pk-p_{k-1} = d.
    # This means we are looking for the maximum k such that there exists i, d 
    # where h[i] = h[i+d] = h[i+2d] = ... = h[i+(k-1)d].
    
    # Correct logic:
    # For every starting index i (0 to N-1)
    # For every interval d (1 to N-1)
    # Count how many consecutive elements starting from i with step d have height h[i].
    # The number of elements is k such that h[i] == h[i+d] == ... == h[i+(k-1)d].
    
    # To implement this without loops:
    # We can use a list comprehension to generate the sequence of heights:
    # [h[i + k*d] for k in range((n-i-1)//d + 1)]
    # Then we need to count the length of the prefix of elements equal to h[i].
    # Since we can't use while, we can use a trick:
    # Find the first index k where h[i + k*d] != h[i].
    # The count is then k.
    
    # To find the first index k where the condition fails:
    # We can use a list comprehension to find all k where h[i+k*d] != h[i]
    # and take the minimum of those k.
    
    # Let's refine:
    # For i in 0..N-1:
    #   For d in 1..N-1:
    #     k_max = (n-1-i)//d + 1
    #     # Find the first k in 0..k_max-1 such that h[i+k*d] != h[i]
    #     # The number of buildings is that k.
    #     # If all are equal, the number is k_max.
    
    # Using a generator expression inside next() to find the first failure:
    # count = next((k for k in range(k_max) if h[i + k*d] != h[i]), k_max)
    
    # Finally, take the max of all these counts.
    
    # Since we can't use for loops, we use nested comprehensions.
    
    ans = max([
        next((k for k in range((n - 1 - i) // d + 1) if h[i + k*d] != h[i]), (n - 1 - i) // d + 1)
        for i in range(n)
        for d in range(1, n)
    ] + [1])
    
    print(ans)

if __name__ == "__main__":
    solve()