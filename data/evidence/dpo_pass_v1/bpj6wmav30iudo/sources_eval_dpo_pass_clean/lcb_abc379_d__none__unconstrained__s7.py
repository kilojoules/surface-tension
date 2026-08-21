import sys
import heapq
from functools import reduce

def solve():
    # Read all input lines
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return
    
    Q = int(input_data[0])
    queries = input_data[1:]

    # State consists of:
    # 1. current_time: Total T elapsed since the start.
    # 2. plants: A min-heap of (planting_time) such that 
    #    height = current_time - planting_time.
    #    Condition height >= H becomes: current_time - planting_time >= H
    #    Which is: planting_time <= current_time - H.
    # 3. results: A list to store the counts for type 3 queries.
    
    # We use a list for state to allow mutation inside the reduce lambda
    # state = [current_time, plants_heap, results_list]
    initial_state = [0, [], []]

    def process_query(state, query_str):
        parts = list(map(int, query_str.split()))
        q_type = parts[0]
        
        # Type 1: Plant a new plant. 
        # Height is 0, so planting_time = current_time.
        # We store -planting_time to simulate a max-heap using heapq (min-heap),
        # because we want to harvest plants with the smallest planting_time (oldest).
        # Actually, since we need planting_time <= current_time - H, 
        # we need the smallest planting_time. A standard min-heap is perfect.
        
        if q_type == 1:
            heapq.heappush(state[1], state[0])
        
        elif q_type == 2:
            # Type 2: Increase time.
            state[0] += parts[1]
            
        elif q_type == 3:
            # Type 3: Harvest plants where planting_time <= current_time - H.
            H = parts[1]
            threshold = state[0] - H
            
            # We need to count how many plants have planting_time <= threshold.
            # Since we cannot use while loops, we use a recursive-like 
            # approach via a helper function? No, recursion is banned.
            # We can use a list comprehension or filter, but we need to 
            # remove elements from the heap.
            
            # To remove elements without a while loop, we can use 
            # a custom function passed to reduce or a clever trick.
            # However, the prompt allows "list comprehensions" and "map/filter".
            # But we need to modify the heap.
            
            # Since we must avoid loops, we can use a recursive-like 
            # structure via a helper that calls itself? No, recursion banned.
            # Let's use a generator and `deque.popleft` or similar? No.
            
            # Wait, I can use a list comprehension to identify which plants 
            # to keep and which to harvest, then re-heapify.
            # harvested_count = len([p for p in state[1] if p <= threshold])
            # state[1] = [p for p in state[1] if p > threshold]
            # heapq.heapify(state[1])
            
            # Let's implement this logic:
            threshold = state[0] - H
            # Identify plants to be harvested
            harvested = [p for p in state[1] if p <= threshold]
            # Keep plants that are too short
            state[1] = [p for p in state[1] if p > threshold]
            # Restore heap property
            heapq.heapify(state[1])
            # Record result
            state[2].append(len(harvested))
            
        return state

    # Process all queries using reduce
    reduce(process_query, queries, initial_state)
    
    # Print results using join and map to avoid for loop
    sys.stdout.write("\n".join(map(str, initial_state[2])) + "\n")

if __name__ == "__main__":
    solve()