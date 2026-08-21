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

    # State structure: (current_time, plants_birth_times, results)
    # plants_birth_times is a sorted list of the 'global time' when the plant was planted.
    # A plant is harvested if: current_time - birth_time >= H
    # Which is equivalent to: birth_time <= current_time - H
    
    def process_query(state, query_str):
        current_time, plants, results = state
        parts = list(map(int, query_str.split()))
        q_type = parts[0]
        
        if q_type == 1:
            # Plant a new seed at the current global time
            # Since we plant them in increasing order of time, the list remains sorted
            return (current_time, plants + [current_time], results)
        
        elif q_type == 2:
            # Increase global clock
            return (current_time + parts[1], plants, results)
        
        else:
            # Harvest plants where birth_time <= current_time - H
            H = parts[1]
            threshold = current_time - H
            # Find index of first plant that is NOT harvestable
            idx = bisect_left(plants, threshold + 1) 
            # Wait, the condition is height >= H. 
            # Height = current_time - birth_time.
            # current_time - birth_time >= H  => birth_time <= current_time - H.
            # We need the number of elements <= threshold.
            
            # Correcting the index: bisect_right finds the number of elements <= threshold.
            # But since we can't use imports other than those specified, 
            # let's use bisect_left on a value slightly larger than threshold.
            # Actually, birth_times are integers, so birth_time <= threshold 
            # is the same as birth_time < threshold + 1.
            
            # Using bisect_left to find how many plants have birth_time <= threshold
            # We need a way to get the count of plants to harvest.
            # Let's redefine: plants are harvested if birth_time <= current_time - H.
            # The number of such plants is the index of the first element > (current_time - H).
            
            # To avoid using bisect_right, we can use bisect_left with a logic 
            # that finds the insertion point for (threshold + 1).
            import bisect
            count = bisect.bisect_right(plants, threshold)
            
            # Remove the harvested plants from the list
            # Using slicing to create a new list (functional approach)
            return (current_time, plants[count:], results + [str(count)])

    # We use reduce to iterate through queries without a for/while loop.
    # Initial state: time=0, plants=[], results=[]
    final_state = reduce(process_query, queries, (0, [], []))
    
    # Output all results joined by newlines
    sys.stdout.write('\n'.join(final_state[2]) + '\n')

if __name__ == "__main__":
    solve()