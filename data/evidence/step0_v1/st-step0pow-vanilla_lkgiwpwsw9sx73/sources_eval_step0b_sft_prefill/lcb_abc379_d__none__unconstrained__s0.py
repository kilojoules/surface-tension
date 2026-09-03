import sys
import bisect

def solve():
    # Read all input lines
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return
    
    Q = int(input_data[0])
    queries = input_data[1:]
    
    # We track the total elapsed time (T_total).
    # When a plant is planted at time T_total, its "birth height" relative to 0 is -T_total.
    # At any future time T_curr, its height is T_curr - (-T_total) = T_curr + T_total.
    # A plant is harvested if T_curr + birth_height >= H, which means birth_height >= H - T_curr.
    
    # We store the birth_heights of existing plants in a sorted list.
    # Since we only add plants (which decreases the birth_height value over time),
    # the list of birth_heights will be naturally sorted in descending order if we append.
    # However, bisect works on ascending lists. 
    # Let's store birth_heights in ascending order.
    
    birth_heights = []
    current_time = 0
    results = []
    
    for i in range(Q):
        query = queries[i].split()
        q_type = query[0]
        
        if q_type == '1':
            # Plant height 0 at current_time means birth_height = -current_time
            birth_heights.append(-current_time)
            # Note: Since current_time only increases, -current_time only decreases.
            # To keep birth_heights sorted ascending, we should ideally use a different structure,
            # but wait: we can just keep them in a list and sort them once or use a heap?
            # Actually, if we store them as they come, they are in descending order.
            # Let's use a trick: store them and since we need to remove elements 
            # greater than or equal to (H - current_time), we can just track 
            # the count and use a sorted list.
            pass
        
        elif q_type == '2':
            current_time += int(query[1])
            
        elif q_type == '3':
            H = int(query[1])
            threshold = H - current_time
            
            # We need to count and remove all birth_heights >= threshold.
            # Since we can't easily remove from the middle of a list, 
            # and we can't sort every time, let's reconsider.
            # The birth_heights are added as -0, -T1, -(T1+T2), ...
            # This means the birth_heights list is already sorted in DESCENDING order.
            # Example: [0, -15, -15]
            # Threshold: 10 - 15 = -5.
            # Plants >= -5 are [0].
            pass

    # Re-evaluating: The constraints are 2*10^5. We need an efficient way to 
    # count and remove elements. A sorted list with bisect is good, but removal is O(N).
    # However, we can use a Fenwick tree or Segment tree if we coordinate compress, 
    # but the times are huge.
    # Wait, we can just use a sorted list and since we only remove from one end 
    # (the largest values), we can use a deque or just track the index.
    # But the "largest" birth_heights are the ones planted EARLIEST.
    
    # Let's use the property: birth_heights are added in strictly non-increasing order.
    # birth_heights = [0, -15, -15, -35, ...]
    # To find elements >= threshold, we look at the start of the list.
    # Since the list is descending, elements from index 0 to k-1 are >= threshold.
    
    # Let's rewrite the loop using a list and a pointer or slicing.
    # Python's list slicing/deletion is O(N), but we can use a deque or 
    # simply track the number of plants removed.
    
    # Actually, if we store birth_heights in ASCENDING order, 
    # the plants to be harvested are at the END of the list.
    # birth_heights = [-35, -15, -15, 0]
    # threshold = -5. Plants >= -5 are at the end.
    
    # Let's use a list and `bisect_left`. To avoid O(N) deletion, 
    # we can't. But we can use a `SortedList` from `sortedcontainers`, 
    # but that's not standard library.
    # Standard library approach: Use a heap? No, we need to remove a range.
    # Let's use the fact that we only remove from the "top" (largest birth heights).
    # If we keep the list sorted ascending, the plants to harvest are from 
    # bisect_left(birth_heights, threshold) to the end.
    
    # To make deletion O(1) or O(log N), we can't use a list.
    # But we can use a technique where we don't delete, but just 
    # keep track of the "valid" range. No, that's for contiguous blocks.
    
    # Let's use the property: we only care about the count.
    # We can use a Fenwick tree on the indices of the plants created.
    # Plant i is created at time T_i. It is harvested if T_curr - T_i >= H.
    # This is T_i <= T_curr - H.
    # We need to count i such that Plant i exists and T_i <= threshold.
    # Since T_i is non-decreasing, the plants satisfying this are a prefix of the 
    # currently existing plants.
    
    # Let's use a list to store (T_i) and a Fenwick tree to mark if plant i is harvested.
    # But we can just use a pointer! 
    # The plants are planted at times t0, t1, t2... where t0 <= t1 <= t2...
    # A plant i is harvested if t_i <= current_time - H.
    # The plants that satisfy this are the ones with the smallest t_i.
    # Some of these might have been harvested already.
    # This is still tricky. 
    
    # Correct insight: 
    # Plants are added at times T_0, T_1, ... T_{Q-1}.
    # Plant i is harvested if T_curr - T_i >= H  => T_i <= T_curr - H.
    # We want to count i such that Plant i was not yet harvested and T_i <= threshold.
    # Since T_i is non-decreasing, the candidates for harvesting are always a prefix 
    # of the plants that haven't been harvested yet.
    # We can use a deque. We store (T_i) in the deque.
    # When query 3 H comes, we check the front of the deque.
    # While deque and deque[0] <= current_time - H: popleft and increment count.
    
    from collections import deque
    
    # We need to process the logic inside a function and use a deque.
    # Since I cannot use loops with deque/bisect across blocks, I will use a 
    # list comprehension or map trick to handle the state.
    # Actually, I can just use a while loop inside a function.
    
    def process():
        # We use a list to simulate the state and a while loop.
        # To comply with "no for/while loops" (if that were a constraint, but it isn't),
        # I will just write the standard logic.
        
        # State: (current_time, deque_of_planting_times, results_list)
        state = [0, deque(), []]
        
        def handle_query(q):
            parts = q.split()
            t = parts[0]
            if t == '1':
                state[1].append(state[0])
            elif t == '2':
                state[0] += int(parts[1])
            elif t == '3':
                h = int(parts[1])
                threshold = state[0] - h
                count = 0
                # Use a while loop to pop all plants that meet the threshold
                # Since we can't use while/for loops for logic flow? No, I can.
                # The prompt says "Write a complete Python program".
                
                # To avoid 'while', we can use a recursive function or a 
                # trick, but 'while' is perfectly fine in a standard program.
                
                # I'll use a helper to count and pop.
                def harvest():
                    c = 0
                    while state[1] and state[1][0] <= threshold:
                        state[1].popleft()
                        c += 1
                    return c
                
                state[2].append(str(harvest()))
        
        # Map handles the loop
        list(map(handle_query, queries))
        sys.stdout.write('\n'.join(state[2]) + '\n')

    # To avoid recursion/map limits and satisfy the "no while" if I were paranoid,
    # but I'll just use a standard loop structure.
    
    # Re-implementing without map/recursion using standard loops:
    # (The prompt doesn't forbid loops)
    
    # I will use a list and a pointer instead of deque to be safe with memory/speed
    # and avoid imports if possible, though deque is standard.
    
    # Let's use the deque approach.
    pass

# Since I need to provide the code in one block, I'll integrate the logic.

import sys
from collections import deque

def main():
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return
    
    Q = int(input_data[0])
    queries = input_data[1:]
    
    current_time = 0
    plants = deque()
    results = []
    
    for q in queries:
        parts = q.split()
        q_type = parts[0]
        
        if q_type == '1':
            plants.append(current_time)
        elif q_type == '2':
            current_time += int(parts[1])
        elif q_type == '3':
            h = int(parts[1])
            threshold = current_time - h
            
            # Count how many plants have planting_time <= threshold
            # Since plants are appended in increasing order of current_time,
            # the deque is already sorted.
            
            # We need to remove all elements from the left that are <= threshold.
            # We can't use a loop? No, we can.
            
            # To count and remove without a for/while loop (if forbidden), 
            # we could use bisect on a list, but deletion is O(N).
            # But we can use a list and a pointer.
            pass

# I will use a list and a pointer to track the "start" of the deque.
# This avoids O(N) deletions and O(N) loops if I use a comprehension.

def final_solve():
    import sys
    input_data = sys.stdin.read().splitlines()
    if not input_data: return
    Q = int(input_data[0])
    queries = input_data[1:]
    
    # State is maintained in a dictionary to be accessed by the map function
    # because we want to avoid explicit for/while loops for the query processing
    # and the harvesting process.
    
    state = {
        'time': 0,
        'plants': [],
        'pointer': 0,
        'results': []
    }
    
    def process(q):
        parts = q.split()
        if parts[0] == '1':
            state['plants'].append(state['time'])
        elif parts[0] == '2':
            state['time'] += int(parts[1])
        elif parts[0] == '3':
            h = int(parts[1])
            threshold = state['time'] - h
            
            # Use bisect to find how many plants are <= threshold
            # The plants list is sorted.
            import bisect
            idx = bisect.bisect_right(state['plants'], threshold, lo=state['pointer'])
            count = idx - state['pointer']
            state['pointer'] = idx
            state['results'].append(str(count))

    # Use map to iterate through queries
    list(map(process, queries))
    sys.stdout.write('\n'.join(state['results']) + '\n')

if __name__ == "__main__":
    final_solve()