import sys

def solve():
    # Read N from the first line of input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # Group indices of buildings by their height
    # height_groups[height] = [index1, index2, ...]
    height_groups = {}
    for i in range(n):
        height = h[i]
        if height not in height_groups:
            height_groups[height] = []
        height_groups[height].append(i)
    
    max_buildings = 0
    
    # For each height group, check all possible intervals
    for height in height_groups:
        indices = height_groups[height]
        num_indices = len(indices)
        
        # If the number of buildings of this height is less than current max, skip
        if num_indices <= max_buildings:
            continue
            
        # Try every pair of buildings of the same height as the start and second element
        # to define the interval 'd'
        for i in range(num_indices):
            start_idx = indices[i]
            for j in range(i + 1, num_indices):
                next_idx = indices[j]
                d = next_idx - start_idx
                
                # Count how many buildings of the same height exist at interval d
                count = 0
                curr = start_idx
                while curr < n:
                    if h[curr] == height:
                        count += 1
                    else:
                        # The condition says "the chosen buildings" must have the same height.
                        # It does NOT say every building at the interval must have that height.
                        # Wait, re-reading: "The chosen buildings all have the same height" 
                        # AND "The chosen buildings are arranged at equal intervals."
                        # This means if we pick indices i, i+d, i+2d..., they must all have height H.
                        # If one doesn't, we can't include it, but the interval is fixed by our choice.
                        # Actually, the most restrictive interpretation is:
                        # We pick a starting index 's' and an interval 'd'.
                        # We pick a subset of indices {s, s+d, s+2d, ... s+kd} such that 
                        # all these indices have height H.
                        # To maximize the number of buildings, we want the longest 
                        # contiguous sequence of the form s + m*d that all have height H.
                        # NO, the problem says "The chosen buildings are arranged at equal intervals."
                        # This implies if we choose k buildings, they must be at s, s+d, ..., s+(k-1)d.
                        # All of these must have the same height.
                        break 
                    curr += d
                
                if count > max_buildings:
                    max_buildings = count
                    
    # Handle the case where N=1 or no pairs were found (max_buildings remains 0)
    # The constraint says if we choose one building, it satisfies the condition.
    if n > 0 and max_buildings == 0:
        max_buildings = 1
        
    # Special case: if we only have buildings of different heights, 
    # the nested loop for j in range(i+1, ...) never executes.
    # But we can always pick 1 building.
    # Let's refine the logic to ensure we check for 1.
    
    # Since the loops above only check intervals d > 0 (2 or more buildings),
    # we initialize max_buildings to 1 if N >= 1.
    
    # Let's rewrite the logic slightly to be more robust.
    print(max(1, max_buildings) if n > 0 else 0)

# The logic inside solve() had a slight flaw regarding the 1-building case.
# Let's provide a cleaner version.
def solve_final():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    if n == 0:
        print(0)
        return

    # Group indices by height
    height_groups = {}
    for i in range(n):
        height = h[i]
        if height not in height_groups:
            height_groups[height] = []
        height_groups[height].append(i)
    
    ans = 1
    for height, indices in height_groups.items():
        m = len(indices)
        if m <= ans:
            continue
        
        # Try all pairs to define the interval d
        for i in range(m):
            for j in range(i + 1, m):
                d = indices[j] - indices[i]
                # We start at indices[i] and check how many consecutive 
                # steps of size d also have the target height.
                count = 0
                curr = indices[i]
                while curr < n and h[curr] == height:
                    count += 1
                    curr += d
                if count > ans:
                    ans = count
    print(ans)

if __name__ == "__main__":
    solve_final()