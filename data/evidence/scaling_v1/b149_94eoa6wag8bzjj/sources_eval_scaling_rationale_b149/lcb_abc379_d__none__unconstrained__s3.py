import sys
from bisect import bisect_left
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We need to process queries in order. 
    # State: (current_total_height, plants_birth_heights, results)
    # plants_birth_heights will store the 'relative' height at which a plant was planted.
    # If a plant is planted when total_height is S, its relative height is -S.
    # Its actual height at any time is: current_total_height + relative_height.
    # Condition: current_total_height + relative_height >= H  =>  relative_height >= H - current_total_height.
    
    # To avoid loops, we use a generator to group the flat input list into queries.
    def get_queries(data):
        it = iter(data[1:])
        def produce():
            try:
                while True:
                    q_type = next(it)
                    if q_type == '1':
                        yield (1, 0)
                    elif q_type == '2':
                        yield (2, int(next(it)))
                    else:
                        yield (3, int(next(it)))
            except StopIteration:
                pass
        return produce()

    def process_query(state, query):
        total_height, plants, results = state
        q_type, val = query
        
        if q_type == 1:
            # Plant height 0. Relative height = 0 - total_height.
            # We maintain 'plants' as a sorted list of relative heights.
            # Since total_height only increases, new relative heights are always smaller.
            # However, to use bisect, we need a sorted list. 
            # Actually, since we only remove from the "top" (largest relative heights),
            # and new plants have the smallest relative heights, the list remains 
            # effectively sorted if we append and then manage the harvest.
            # Wait, new plants have relative height -total_height. 
            # As total_height increases, -total_height decreases.
            # So new plants are always added to the left of the sorted list.
            # To keep it O(log N), we store relative heights and use bisect.
            # Since we add to the left, we can store them as is and use insort or 
            # just realize that the relative heights are added in non-increasing order.
            # Let's use a list and bisect.insort is O(N), but we can store them 
            # and sort only when needed? No, that's O(N log N).
            # Actually, the relative heights are added in strictly non-increasing order.
            # Let's store them and use the fact that they are added at the "bottom".
            # To keep it sorted, we can store them and use the fact that 
            # the list of relative heights is always sorted if we insert at index 0.
            # But insert(0) is O(N). 
            # Let's store them as they come and use a sorted list. 
            # Given Q=2e5, O(N) insertions will TLE.
            # Wait, the relative heights are added as -total_height.
            # Since total_height is non-decreasing, -total_height is non-increasing.
            # If we store them in a list, the list is sorted in descending order.
            # To use bisect, we want ascending. So we can store them and 
            # since they are added in descending order, we can just append and 
            # the list is "reverse sorted".
            # Let's just use a standard list and bisect.insort. 
            # Actually, the most efficient way is to realize that 
            # relative heights are added in non-increasing order.
            # If we store them in a list, the list is always sorted if we 
            # treat it as a stack and the harvest removes from the "end" 
            # if we negate the values.
            # Let's use: relative_height = total_height - 0.
            # Height = total_height - relative_height.
            # Harvest if: total_height - relative_height >= H  => relative_height <= total_height - H.
            # New plants have relative_height = total_height.
            # Since total_height increases, new relative_heights are larger.
            # The list of relative_heights is naturally sorted!
            return (total_height, plants + [total_height], results)
            
        elif q_type == 2:
            return (total_height + val, plants, results)
            
        elif q_type == 3:
            # Harvest if relative_height <= total_height - val
            threshold = total_height - val
            # Find index of first element > threshold
            idx = bisect_left(plants, threshold + 0.1) # Using 0.1 to simulate >
            # Wait, bisect_left finds the first index where element >= threshold.
            # We want elements <= threshold.
            # Those are indices 0 to bisect_right(plants, threshold) - 1.
            import bisect
            count_idx = bisect.bisect_right(plants, threshold)
            harvested_count = count_idx
            # Remove the first 'harvested_count' elements.
            # Slicing is O(N). To avoid O(N), we can't easily remove from the front.
            # But we can keep track of an offset.
            # Let's redefine state: (total_height, plants, offset, results)
            return state # This structure needs updating.

    # Redefining the logic to avoid O(N) operations and loops.
    # State: (total_height, plants_list, offset, results)
    def process_final(state, query):
        total_height, plants, offset, results = state
        q_type, val = query
        if q_type == 1:
            return (total_height, plants + [total_height], offset, results)
        if q_type == 2:
            return (total_height + val, plants, offset, results)
        if q_type == 3:
            import bisect
            threshold = total_height - val
            # We are looking for plants[i] <= threshold for i in [offset, len(plants))
            # Since plants is sorted, we find the upper bound of threshold.
            upper_idx = bisect.bisect_right(plants, threshold, lo=offset)
            count = upper_idx - offset
            results.append(str(count))
            return (total_height, plants, upper_idx, results)

    # Use reduce to simulate the loop
    final_state = reduce(process_final, get_queries(input_data), (0, [], 0, []))
    sys.stdout.write("\n".join(final_state[3]) + "\n")

if __name__ == "__main__":
    solve()