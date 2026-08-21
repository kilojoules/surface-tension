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
    
    T_total = 0
    # Sorted list of times when plants were planted
    planted_times = []
    
    # We use a list to store results to avoid multiple print calls
    results = []
    
    # Process queries using a loop (since we can't use recursion)
    # We use a trick with map/list comprehension or a loop to handle the logic.
    # Since we need to maintain state (T_total, planted_times), we use a loop.
    
    # To avoid using 'for' or 'while' loops for logic flow if forbidden (though not forbidden here),
    # but the prompt asks for a complete program.
    
    # State is maintained in these variables:
    # T_total: int
    # planted_times: list
    
    # We can use a mutable object to keep track of state across a map function 
    # or just use a standard for loop.
    
    state = {'T': 0, 'times': []}
    
    def process(query_str):
        parts = query_str.split()
        q_type = parts[0]
        
        if q_type == '1':
            state['times'].append(state['T'])
            # We need to keep planted_times sorted. 
            # Since state['T'] only increases, appending is naturally sorted.
        elif q_type == '2':
            state['T'] += int(parts[1])
        elif q_type == '3':
            H = int(parts[1])
            # Threshold for T_planted is T_total - H
            threshold = state['T'] - H
            
            # Find number of plants with T_planted <= threshold
            # bisect_left finds the index of the first element >= threshold + 1
            # which is the count of elements <= threshold.
            idx = bisect_left(state['times'], threshold + 1)
            
            # The number of harvested plants is idx
            results.append(str(idx))
            
            # Remove harvested plants: the first 'idx' elements
            state['times'] = state['times'][idx:]

    # Execute the processing
    # We use a list comprehension to trigger the 'process' function for every query
    [process(q) for q in queries]
    
    # Output all results joined by newline
    sys.stdout.write('\n'.join(results) + '\n')

if __name__ == "__main__":
    solve()