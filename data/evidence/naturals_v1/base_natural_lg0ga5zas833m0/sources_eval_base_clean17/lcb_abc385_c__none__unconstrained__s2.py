import sys

def solve():
    # Read N from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # We want to find the maximum number of buildings with the same height
    # that are placed at equal intervals.
    # A set of buildings at equal intervals is defined by:
    # starting index 'i' and step size 'd'.
    # The indices are i, i+d, i+2d, ...
    
    max_buildings = 0
    
    # If N is 0, the answer is 0. Otherwise, at least 1 building can be chosen.
    if N > 0:
        max_buildings = 1
    else:
        print(0)
        return

    # Iterate through all possible starting positions i
    for i in range(N):
        # Iterate through all possible step sizes d
        # The maximum possible step size is N-1
        for d in range(1, N):
            # To avoid redundant checks, we only check if the starting building 
            # can potentially beat the current max_buildings.
            # However, with N=3000, O(N^3) is too slow. 
            # We need to optimize.
            
            # Optimization: only check if the building at i and i+d have the same height.
            if i + d >= N:
                break
                
            if H[i] == H[i + d]:
                count = 0
                # Check how many buildings in this sequence have the same height H[i]
                # The condition is that ALL chosen buildings must have the same height.
                # The problem says "The chosen buildings all have the same height" 
                # and "are arranged at equal intervals".
                # This implies we are looking for a sequence (i, i+d, i+2d...)
                # where every element in that sequence has height H[i].
                
                # Wait, the problem doesn't say we must pick ALL buildings in the 
                # arithmetic progression, but that the ones we CHOOSE must be at 
                # equal intervals. This means we are looking for a subset of indices 
                # {i, i+d, i+2d, ..., i+(k-1)d} such that H[i] = H[i+d] = ... = H[i+(k-1)d].
                
                # Let's re-evaluate:
                # For a fixed start i and step d, we count how many j = i + k*d 
                # have H[j] == H[i].
                # But the "equal intervals" condition means the distance between 
                # consecutive chosen buildings must be the same.
                # So if we pick indices p1, p2, ..., pk, then p2-p1 = p3-p2 = ... = d.
                
                # This means we are indeed looking for the longest sequence 
                # i, i+d, i+2d... where all have the same height.
                # However, the "equal intervals" could be any d.
                # If we find a sequence with height H[i] and step d, 
                # we just need to check how many consecutive terms match.
                
                # Actually, the simplest interpretation is:
                # Pick a height 'h', a starting position 'i', and a step 'd'.
                # Count how many j = i + k*d satisfy H[j] == h.
                # But the indices must be i, i+d, i+2d... 
                # If H[i+d] is not h, we can't just skip it and pick H[i+2d] 
                # because then the interval between the 1st and 2nd chosen 
                # building would be 2d, not d.
                # Wait, if we pick i and i+2d, the interval is 2d. That's still "equal intervals".
                # So we are looking for the maximum k such that there exists i, d 
                # where H[i] = H[i+d] = H[i+2d] = ... = H[i+(k-1)d].
                
                # Let's refine the loop:
                current_count = 0
                for j in range(i, N, d):
                    if H[j] == H[i]:
                        current_count += 1
                    else:
                        # Since the chosen buildings must be at equal intervals,
                        # if we encounter a building with a different height,
                        # we can't just "skip" it and keep the same d.
                        # If we skip it, the interval changes.
                        # Actually, the problem says "The chosen buildings are arranged at equal intervals".
                        # This means the indices are a, a+d, a+2d...
                        # It does NOT say that all buildings at those indices must be chosen.
                        # It says the ones we CHOOSE must be at equal intervals.
                        # This means if we choose indices {p1, p2, ..., pk}, 
                        # then p2-p1 = p3-p2 = ... = d.
                        # This is exactly what I wrote: H[i] = H[i+d] = H[i+2d]...
                        # But if H[i+d] != H[i], we can't include it.
                        # If we don't include it, we can't include H[i+2d] either 
                        # because the distance from the previous chosen building (H[i]) 
                        # would be 2d, while the distance from the next would be d.
                        # Therefore, the chosen buildings must form an 
                        # arithmetic progression of indices where all have the same height.
                        
                        # Wait, if we skip H[i+d], the sequence is i, i+2d, i+3d... 
                        # That is NOT equal intervals.
                        # The only way to have equal intervals is to have a 
                        # constant difference d between all consecutive chosen indices.
                        # So if we choose i, i+d, i+2d... then H[i], H[i+d], H[i+2d]... 
                        # must all be the same height.
                        
                        # If H[i+d] != H[i], then we cannot have a sequence 
                        # starting at i with step d and length > 1.
                        # Because the second element must be i+d.
                        break 
                
                if current_count > max_buildings:
                    max_buildings = current_count

    print(max_buildings)

# The O(N^3) approach might be too slow for N=3000.
# Let's optimize.
def solve_optimized():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    if N == 0:
        print(0)
        return
    
    ans = 1
    # Group indices by height
    from collections import defaultdict
    pos_map = defaultdict(list)
    for i in range(N):
        pos_map[H[i]].append(i)
        
    for height in pos_map:
        indices = pos_map[height]
        m = len(indices)
        if m <= ans:
            continue
            
        # For each pair of indices (i, j) with the same height, 
        # they define a potential interval d = j - i.
        for a in range(m):
            for b in range(a + 1, m):
                i = indices[a]
                j = indices[b]
                d = j - i
                
                # We check how many more buildings with the same height 
                # exist at intervals of d.
                count = 2
                next_idx = j + d
                # We only need to check indices that are in the pos_map for this height
                # Using a set for O(1) lookup
                # But creating a set every time is slow. 
                # Let's use a boolean array or a set created once per height.
                pass
    
    # Let's rewrite the logic more cleanly.

def final_solve():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    if N == 0:
        print(0)
        return
    
    # Use a boolean array for fast lookup of heights
    # Since N is 3000, we can afford some O(N^2)
    # For each possible step d from 1 to N:
    #   For each starting position i from 0 to N-1:
    #     Check the sequence i, i+d, i+2d...
    
    # Actually, the most efficient way:
    # For each height that appears in the array:
    #   Find all indices where that height occurs.
    #   For every pair of these indices (i, j), d = j - i.
    #   Check how many k = j + d, j + 2d... also have that height.
    
    from collections import defaultdict
    pos_map = defaultdict(list)
    for i in range(N):
        pos_map[H[i]].append(i)
    
    max_count = 1
    for height, indices in pos_map.items():
        if len(indices) <= max_count:
            continue
        
        # Use a set for O(1) lookup of indices with the same height
        idx_set = set(indices)
        
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                i = indices[a]
                j = indices[b]
                d = j - i
                
                # Potential count is (len(indices) - a) // 1... not really.
                # We can prune: if current count + remaining indices < max_count, stop.
                # But we don't know the remaining. 
                # We can check: if (N - 1 - j) // d + 2 <= max_count: continue
                if (N - 1 - j) // d + 2 <= max_count:
                    continue
                
                count = 2
                curr = j + d
                while curr < N and curr in idx_set:
                    count += 1
                    curr += d
                
                if count > max_count:
                    max_count = count
                    
    print(max_count)

if __name__ == "__main__":
    final_solve()