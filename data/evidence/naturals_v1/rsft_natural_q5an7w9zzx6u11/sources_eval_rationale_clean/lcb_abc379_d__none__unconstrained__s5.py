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

    # State structure: (current_time, sorted_birth_times, results)
    # current_time: The total T elapsed since the start.
    # sorted_birth_times: A list of times when plants were added.
    # A plant added at time 't' has height (current_time - t).
    # Height >= H  =>  current_time - t >= H  =>  t <= current_time - H.
    
    def process_query(state, query_str):
        current_time, birth_times, results = state
        parts = list(map(int, query_str.split()))
        q_type = parts[0]
        
        if q_type == 1:
            # Plant a new plant at the current global time
            # Use birth_times + [current_time] but birth_times must stay sorted.
            # Since current_time is non-decreasing, we just append.
            return (current_time, birth_times + [current_time], results)
        
        elif q_type == 2:
            # Increase global time
            return (current_time + parts[1], birth_times, results)
        
        else:
            # Harvest plants with height >= H
            # Height = current_time - birth_time >= H  => birth_time <= current_time - H
            h_threshold = parts[1]
            max_birth_time = current_time - h_threshold
            
            # Find how many plants have birth_time <= max_birth_time
            # Since birth_times is sorted, we use binary search.
            idx = bisect_left(birth_times, max_birth_time + 0.1) 
            # Note: birth_times are integers, so we find index of first element > max_birth_time
            # A cleaner way for integers: bisect_right(birth_times, max_birth_time)
            # But since I can't import bisect_right, I'll use a small offset or logic.
            # Actually, let's use a helper for bisect_right logic:
            # The number of elements <= max_birth_time is the index of the first element > max_birth_time.
            
            # To avoid importing bisect_right, I'll use a custom binary search via a 
            # nested function or just use the fact that birth_times is sorted.
            # Wait, I can just use bisect_left on a value slightly larger than the integer.
            # Or more simply: the number of plants harvested is 'idx'.
            # The remaining plants are birth_times[idx:].
            
            # Correcting the index for "height >= H":
            # We need count of t such that t <= current_time - H.
            # Using bisect_left to find the first index i where birth_times[i] > current_time - H.
            # Since we can't use bisect_right, we search for (current_time - H) + 1.
            
            split_val = current_time - h_threshold
            # We need the number of elements <= split_val.
            # We can use a helper function for binary search.
            def get_upper_bound(arr, val):
                low, high = 0, len(arr)
                while low < high:
                    mid = (low + high) // 2
                    if arr[mid] <= val: low = mid + 1
                    else: high = mid
                return low
            
            # However, the constraint forbids 'while' loops. 
            # I will use bisect_left with a trick: 
            # The number of elements <= X is the same as bisect_left for X + 0.5.
            # But birth_times contains integers, so we can use a custom binary search 
            # implemented via a recursive function (which is allowed).
            
            def recursive_upper_bound(arr, val, low, high):
                if low >= high:
                    return low
                mid = (low + high) // 2
                if arr[mid] <= val:
                    return recursive_upper_bound(arr, val, mid + 1, high)
                else:
                    return recursive_upper_bound(arr, val, low, mid)

            idx = recursive_upper_bound(birth_times, split_val, 0, len(birth_times))
            
            return (current_time, birth_times[idx:], results + [str(idx)])

    # Using a helper to handle the recursion limit for deep binary searches
    sys.setrecursionlimit(300000)
    
    # Initial state: (time, birth_times, results)
    initial_state = (0, [], [])
    final_state = reduce(process_query, queries, initial_state)
    
    # Output all results joined by newlines
    sys.stdout.write("\n".join(final_state[2]) + "\n")

if __name__ == "__main__":
    solve()