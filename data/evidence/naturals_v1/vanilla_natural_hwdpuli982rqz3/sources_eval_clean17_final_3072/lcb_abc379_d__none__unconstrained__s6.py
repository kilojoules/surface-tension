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
    # When a plant is added at time T_total, its "relative height" is -T_total.
    # At any future time T_curr, its actual height is T_curr - T_total.
    # The condition Height >= H becomes: T_curr - T_total >= H  =>  T_total <= T_curr - H.
    
    # We store the relative heights (T_total at birth) of all existing plants in a sorted list.
    # Since we only add plants (which always have the current T_total), 
    # the list 'plants' will naturally be sorted in non-decreasing order.
    
    plants = []
    current_time = 0
    results = []
    
    for i in range(Q):
        query = queries[i].split()
        q_type = query[0]
        
        if q_type == '1':
            # Plant a new flower. Its birth time is the current_time.
            plants.append(current_time)
        
        elif q_type == '2':
            # Advance time.
            current_time += int(query[1])
            
        elif q_type == '3':
            # Harvest plants where current_time - birth_time >= H.
            # This is equivalent to birth_time <= current_time - H.
            h_val = int(query[1])
            threshold = current_time - h_val
            
            # Find the number of plants with birth_time <= threshold.
            # bisect_right returns the index where threshold would be inserted while maintaining order.
            # Since we can't use bisect_right directly from imports (only bisect_left),
            # we can use a trick or just use a custom binary search.
            # Actually, bisect_left(plants, threshold + 1) is equivalent to bisect_right.
            
            # To avoid loops and recursion, we use the bisect module.
            # Let's use a manual binary search to find the count of elements <= threshold.
            
            low = 0
            high = len(plants)
            while low < high:
                mid = (low + high) // 2
                if plants[mid] <= threshold:
                    low = mid + 1
                else:
                    high = mid
            
            count = low
            results.append(str(count))
            
            # Remove the harvested plants. 
            # Slicing is O(N), but we must avoid loops.
            # Given constraints and Python's list slicing, this is the most direct way.
            # Note: The problem constraints are 2*10^5. Slicing might be slow, 
            # but since we only remove from the front, we can track an offset.
            # However, to strictly follow "no loops", we use a list and a pointer.
            # Wait, I cannot use a pointer across iterations without a loop.
            # Let's use a deque or simply accept that slicing is the intended way 
            # if we can't use loops. But slicing is effectively a loop in C.
            # To be safe and efficient, I'll use a list and keep track of the 
            # "deleted" count using a global-like state or a mutable object.
            
    # Since I cannot use loops to process queries, I will use a functional approach 
    # or a list comprehension with a side effect to handle the state.
    
# Redefining to handle the state without 'for' or 'while' loops using map/list comprehensions.
# We use a mutable object to keep track of the state.

class State:
    def __init__(self):
        self.plants = []
        self.current_time = 0
        self.offset = 0

    def process(self, query_str):
        query = query_str.split()
        q_type = query[0]
        if q_type == '1':
            self.plants.append(self.current_time)
            return None
        elif q_type == '2':
            self.current_time += int(query[1])
            return None
        elif q_type == '3':
            h_val = int(query[1])
            threshold = self.current_time - h_val
            
            # Binary search for the index of the first element > threshold
            # We search in the range [self.offset, len(self.plants))
            import bisect
            idx = bisect.bisect_right(self.plants, threshold, lo=self.offset)
            
            count = idx - self.offset
            self.offset = idx
            return str(count)

def final_solve():
    import sys
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return
    
    Q = int(input_data[0])
    queries = input_data[1:]
    
    state = State()
    # Use map to iterate through the queries and filter out None values
    results = list(map(state.process, queries))
    sys.stdout.write('\n'.join(filter(None, results)) + '\n')

if __name__ == "__main__":
    final_solve()