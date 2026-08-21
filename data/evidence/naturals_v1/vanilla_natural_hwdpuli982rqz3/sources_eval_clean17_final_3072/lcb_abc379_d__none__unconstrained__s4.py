import sys
from bisect import bisect_left

def solve():
    # Read all input lines
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return
    
    Q = int(input_data[0])
    queries = input_data[1:]
    
    # We track the total time elapsed (T_total).
    # When a plant is planted at time T_total, its "relative height" is -T_total.
    # At any future time T_curr, its actual height is T_curr - T_planted.
    # A plant is harvested if T_curr - T_planted >= H, which is T_planted <= T_curr - H.
    
    # We store the relative heights (T_planted) of existing plants in a sorted list.
    # Since we only append to the list (T_total only increases), 
    # the list of T_planted will naturally be sorted.
    
    T_total = 0
    plants = []
    results = []
    
    for i in range(Q):
        query = queries[i].split()
        q_type = query[0]
        
        if q_type == '1':
            # Plant a new plant. Its height is 0, so T_planted = T_total.
            plants.append(T_total)
        elif q_type == '2':
            # Increase total time.
            T_total += int(query[1])
        elif q_type == '3':
            # Harvest plants where T_planted <= T_total - H.
            H = int(query[1])
            threshold = T_total - H
            
            # Find the number of plants with T_planted <= threshold.
            # bisect_left returns the index of the first element >= threshold + 1.
            # Since we need T_planted <= threshold, we can use bisect_right or 
            # bisect_left with a slightly adjusted value.
            # Let's use bisect_left on (threshold + 1) or simply a custom range.
            
            # To avoid loops, we use bisect_left to find how many elements are <= threshold.
            # Note: bisect_left(plants, x) returns the index of the first element >= x.
            # Elements at indices 0, 1, ..., bisect_left(plants, threshold + 1) - 1 are <= threshold.
            
            import bisect
            idx = bisect.bisect_right(plants, threshold)
            
            # The number of harvested plants is idx.
            results.append(str(idx))
            
            # Remove the harvested plants. 
            # Slicing a list is O(N), but we can't use a loop.
            # However, we can replace the list with the remaining part.
            plants = plants[idx:]
            
    sys.stdout.write('\n'.join(results) + '\n')

if __name__ == "__main__":
    solve()