import sys
from bisect import bisect_left
from functools import reduce

def solve():
    # Read all input lines
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return
    
    Q = int(input_data[0])
    queries = input_data[1:]

    # State consists of:
    # 1. sorted_heights: A list of heights of plants, stored as (H_actual - current_offset)
    # 2. current_offset: The total T accumulated from type 2 queries
    # 3. results: A list to store the answers for type 3 queries
    
    def process_query(state, query_str):
        sorted_heights, current_offset, results = state
        parts = list(map(int, query_str.split()))
        q_type = parts[0]

        if q_type == 1:
            # Plant height is 0. Relative height = 0 - current_offset
            # We use a list and maintain it sorted. 
            # Since we only add '0 - current_offset', and current_offset increases,
            # new plants are always added to the left or we use bisect.
            # However, since we need to avoid loops, we use a list and 
            # we can't use a While loop to pop. 
            # We use slicing to remove elements.
            
            # To keep it sorted: new_val = -current_offset
            # Since current_offset is non-decreasing, -current_offset is non-increasing.
            # New plants are always <= existing plants' relative heights.
            return ([ -current_offset ] + sorted_heights, current_offset, results)

        elif q_type == 2:
            # Increase all heights by T
            return (sorted_heights, current_offset + parts[1], results)

        elif q_type == 3:
            # Harvest plants where height >= H
            # Actual height = relative_height + current_offset
            # relative_height + current_offset >= H  =>  relative_height >= H - current_offset
            H = parts[1]
            threshold = H - current_offset
            
            # Find index of first element >= threshold
            # Note: our list is sorted descending because of how we add plants? 
            # Wait, if we add -current_offset to the front, and current_offset increases,
            # the list remains sorted in descending order.
            # Let's maintain it in ascending order for bisect_left.
            # Correction: Add to the end and sort? No, that's O(N log N).
            # Let's use a different approach for Type 1: 
            # Since -current_offset is non-increasing, we insert at the beginning.
            # That means the list is sorted DESCENDING.
            # For a descending list, bisect doesn't work directly.
            # Let', instead, store relative heights and keep them sorted ASCENDING.
            # New plant relative height: -current_offset.
            # Since -current_offset decreases over time, new plants are always 
            # smaller than or equal to previous plants.
            # So we insert at index 0.
            
            # Actually, if we insert at 0, the list is always sorted ASCENDING.
            # Example: 
            # Q1: T=0, plant 0. List: [0]
            # Q2: T=15. Offset=15.
            # Q3: plant 0. Rel = 0-15 = -15. List: [-15, 0]
            # Q4: H=10. Threshold = 10-15 = -5. 
            # bisect_left([-15, 0], -5) returns index 1.
            # Plants from index 1 to end are harvested.
            
            idx = bisect_left(sorted_heights, threshold)
            harvested_count = len(sorted_heights) - idx
            # Remove harvested plants using slicing
            return (sorted_heights[:idx], current_offset, results + [harvested_count])

    # We need to fix the Type 1 logic to ensure the list is always sorted ascending.
    # Since -current_offset is non-increasing, inserting at the front 
    # would make it descending. We should insert at the front only if 
    # we want descending. For ascending, we insert at the front 
    # ONLY IF the new value is smaller than the current smallest.
    # Since -current_offset is non-increasing, the new plant is ALWAYS 
    # the new minimum. So inserting at index 0 maintains ascending order.
    
    # Redefining process_query slightly to ensure logic is robust
    def process_final(state, query_str):
        sorted_heights, current_offset, results = state
        parts = list(map(int, query_str.split()))
        if parts[0] == 1:
            return ([ -current_offset ] + sorted_heights, current_offset, results)
        if parts[0] == 2:
            return (sorted_heights, current_offset + parts[1], results)
        if parts[0] == 3:
            threshold = parts[1] - current_offset
            idx = bisect_left(sorted_heights, threshold)
            return (sorted_heights[:idx], current_offset, results + [len(sorted_heights) - idx])

    final_state = reduce(process_final, queries, ([], 0, []))
    
    # Output results
    sys.stdout, _ = (sys.stdout.write('\n'.join(map(str, final_state[2])) + '\n'), None)

if __name__ == "__main__":
    solve()