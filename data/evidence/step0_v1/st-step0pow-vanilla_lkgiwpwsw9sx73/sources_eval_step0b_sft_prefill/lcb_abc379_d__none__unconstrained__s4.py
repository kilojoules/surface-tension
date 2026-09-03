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
    # At any future time T_now, its actual height is T_now - T_planted.
    # A plant is harvested if T_now - T_planted >= H, which is T_planted <= T_now - H.
    
    # We store the relative heights (T_planted) of existing plants in a sorted list.
    # Since we only add plants (which always have the current T_total as their T_planted),
    # the list remains sorted if we just append.
    
    T_total = 0
    plants = []
    results = []
    
    # Using a list and bisect is efficient for counting.
    # However, removing elements from the middle of a list is O(N).
    # To avoid O(N) removals, we can maintain a pointer or use a different structure.
    # Given the constraints and the nature of the problem (we remove the smallest T_planted),
    # we can use a deque or simply track the index of the first non-harvested plant.
    # But wait, the condition is T_planted <= T_now - H. 
    # Since T_planted is monotonically increasing (each new plant is planted at the current T_total),
    # the plants that satisfy the condition are always a prefix of the current plants list.
    
    # We use a list to store T_planted and an integer to track the start of the active plants.
    # Since we can't use loops, we use slicing or map/filter? No, slicing is O(N).
    # Let's use a list and keep track of the 'offset' of harvested plants.
    # Actually, the most efficient way to handle "remove prefix" in Python without loops 
    # is to use a list and track the index, but since we need to "output the number",
    # we can calculate the index using bisect_left and then slice the list.
    # While slicing is O(N), we can minimize it. 
    # But the constraint is 2*10^5. O(N^2) will TLE.
    
    # To truly avoid loops and recursion, and avoid O(N) deletions:
    # We can use a list and instead of deleting, we just move the start pointer.
    # But we can't use a while loop to process queries.
    # We can use a functional approach with `reduce` from functools.
    
    from functools import reduce
    from bisect import bisect_right

    def process(state, query):
        T_total, plants, results = state
        parts = query.split()
        q_type = parts[0]
        
        if q_type == '1':
            # Plant new: relative height is current T_total
            return (T_total, plants + [T_total], results)
        
        elif q_type == '2':
            # Wait T days
            T_wait = int(parts[1])
            return (T_total + T_wait, plants, results)
        
        elif q_type == '3':
            # Harvest height H: T_planted <= T_total - H
            H = int(parts[1])
            threshold = T_total - H
            # Find how many plants have T_planted <= threshold
            # bisect_right returns the index of the first element > threshold
            idx = bisect_right(plants, threshold)
            # The number of harvested plants is idx.
            # The remaining plants are plants[idx:]
            return (T_total, plants[idx:], results + [str(idx)])

    # Initial state: (T_total, plants_list, results_list)
    final_state = reduce(process, queries, (0, [], []))
    
    # Output all results joined by newline
    sys.stdout.write("\n".join(final_state[2]) + "\n")

if __name__ == "__main__":
    solve()