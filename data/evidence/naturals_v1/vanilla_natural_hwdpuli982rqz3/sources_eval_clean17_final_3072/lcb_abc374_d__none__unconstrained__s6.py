import sys
import math
from itertools import permutations, product

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    coords = []
    for i in range(N):
        # Each segment is (A, B, C, D)
        coords.append(tuple(map(int, input_data[3 + i*4 : 7 + i*4])))

    # Precompute lengths of segments
    # length[i] = distance between (A_i, B_i) and (C_i, D_i)
    lengths = [math.sqrt((c[0]-c[2])**2 + (c[1]-c[3])**2) for c in coords]
    
    # Helper to get distance between two points
    def dist(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # We need to decide the order of segments and the direction of each segment.
    # N is small (up to 6), so we can iterate through all permutations (N!) 
    # and all direction combinations (2^N).
    
    # Generate all permutations of segment indices
    perms = permutations(range(N))
    # Generate all combinations of directions (0 or 1)
    # 0: (A, B) -> (C, D), 1: (C, D) -> (A, B)
    dirs = product([0, 1], repeat=N)
    
    # To avoid nested loops and recursion, we use a generator/comprehension
    # We calculate the total time for each configuration:
    # Time = (dist(start, p1)/S) + (len1/T) + (dist(p2, p3)/S) + (len2/T) ...
    
    # Since we can't use for-loops, we use a map/min structure.
    # We need to evaluate the cost for every (perm, dir_combo).
    
    def calculate_cost(perm, d_combo):
        # Current position
        curr_x, curr_y = 0, 0
        total_time = 0.0
        
        # We need to simulate the process. Since we can't use for-loops, 
        # we can use a reduce-like approach or a list comprehension with a state.
        # However, the prompt says "Return only Python source", not "no for loops".
        # The "no for loops" constraint is usually for specific functional challenges.
        # I will use standard loops as they are the most readable and efficient way to 
        # handle the state (current position) across the sequence.
        return None # Placeholder

    # Re-evaluating: The prompt does NOT forbid for-loops. It asks for a complete program.
    
    # To find the minimum, we can use a list comprehension to calculate all costs
    # but we need to track the current position. 
    # Let's use a helper function with a loop and call it via map or similar if I were avoiding loops,
    # but I will use standard loops inside a function.
    
    def get_total_time(perm, d_combo):
        time = 0.0
        cx, cy = 0, 0
        for i in range(N):
            idx = perm[i]
            # Endpoints of the segment
            p1 = (coords[idx][0], coords[idx][1])
            p2 = (coords[idx][2], coords[idx][3])
            
            # Determine start and end based on direction
            start_p = p1 if d_combo[i] == 0 else p2
            end_p = p2 if d_combo[i] == 0 else p1
            
            # Move to start (speed S), then print to end (speed T)
            time += dist((cx, cy), start_p) / S
            time += lengths[idx] / T
            cx, cy = end_p
            
        return time

    # Use a generator to save memory and min() to find the minimum
    ans = min(
        get_total_time(p, d) 
        for p in permutations(range(N)) 
        for d in product([0, 1], repeat=N)
    )
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()