import sys

def solve():
    # Read N from the first line of input
    try:
        line1 = sys.stdin.readline()
        if not line1:
            return
        n = int(line1.strip())
        
        # Read H_i from the following lines
        h = []
        while len(h) < n:
            line = sys.stdin.readline()
            if not line:
                break
            h.extend(map(int, line.split()))
    except ValueError:
        return

    # We need to find the maximum number of buildings that have the same height
    # and are placed at equal intervals.
    # Let's group indices by their height.
    height_map = {}
    for i in range(n):
        height = h[i]
        if height not in height_map:
            height_map[height] = []
        height_map[height].append(i)

    max_buildings = 0

    # For each unique height, we check all possible intervals
    for height in height_map:
        indices = height_map[height]
        num_indices = len(indices)
        
        # If the total number of buildings of this height is less than 
        # current max_buildings, we can't possibly beat the record.
        if num_indices <= max_buildings:
            continue
            
        # Try every pair of buildings of the same height as the starting 
        # point and the second point to define the interval 'd'.
        for i in range(num_indices):
            start_idx = indices[i]
            for j in range(i + 1, num_indices):
                # Interval d is the distance between two buildings of the same height
                d = indices[j] - start_idx
                
                # Count how many buildings of the same height exist at this interval
                count = 0
                curr = start_idx
                while curr < n:
                    if h[curr] == height:
                        count += 1
                    else:
                        # The condition says "the chosen buildings all have the same height"
                        # and "are arranged at equal intervals".
                        # If we encounter a building at the interval that doesn't match
                        # the height, we cannot include it in the set.
                        # However, the problem asks for the maximum number we CAN choose.
                        # This means we only count the ones that match.
                        # Wait, the condition is: "The chosen buildings all have the same height"
                        # and "The chosen buildings are arranged at equal intervals".
                        # This means if we pick indices (p, p+d, p+2d...), ALL must have height H.
                        # If one doesn't, we can't include it. But we can just stop 
                        # or skip it? No, "equal intervals" implies a fixed step.
                        # If we pick indices i_1, i_2, ..., i_k, then i_{j+1} - i_j = d.
                        # So we check indices: start, start + d, start + 2d...
                        # If h[curr] != height, that specific index cannot be part of the set.
                        # Since we need them to be at equal intervals, if we skip one, 
                        # the interval is no longer d.
                        # Therefore, the sequence must be contiguous in terms of the interval d.
                        break 
                    curr += d
                
                if count > max_buildings:
                    max_buildings = count

    # If max_buildings is still 0 but N > 0, the answer is 1 (as per problem statement)
    if max_buildings == 0 and n > 0:
        max_buildings = 1
        
    print(max_buildings)

if __name__ == "__main__":
    solve()