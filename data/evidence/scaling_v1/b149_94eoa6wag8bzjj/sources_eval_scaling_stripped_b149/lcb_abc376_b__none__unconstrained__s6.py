import sys

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Parse instructions into a list of (H, T)
    instructions = [
        (input_data[i*2 + 2], int(input_data[i*2 + 3]))
        for i in range(Q)
    ]
    
    # Initial positions
    curr_l = 1
    curr_r = 2
    total_ops = 0
    
    # Helper to calculate distance between two points on a ring of size N
    # without passing through a forbidden point 'p'
    def get_dist(start, end, p):
        # The ring is 1-indexed. We can think of it as 0 to N-1 for easier modulo.
        # However, since we can't pass through p, we are essentially on a line
        # if we "cut" the ring at p.
        
        # Normalize to 0-indexed
        s, e, forbidden = start - 1, end - 1, p - 1
        
        # We can move in two directions: clockwise and counter-clockwise.
        # One direction will be blocked by the forbidden hand.
        # The only way to get from s to e without hitting 'forbidden' is to
        # move in the direction that doesn't contain 'forbidden' in the arc.
        
        # Distance clockwise from s to e
        dist_cw = (e - s) % N
        # Check if forbidden is in the clockwise arc from s to e
        # Forbidden is in arc if (forbidden - s) % N < dist_cw
        blocked_cw = (forbidden - s) % N < dist_cw
        
        # Distance counter-clockwise from s to e
        dist_ccw = (s - e) % N
        # Check if forbidden is in the counter-clockwise arc from s to e
        blocked_ccw = (forbidden - e) % N < dist_ccw
        
        # It is guaranteed that the instruction is achievable.
        # If clockwise is blocked, we must go counter-clockwise, and vice versa.
        # Note: if s == e, distance is 0.
        if s == e: return 0
        
        # We return the distance of the path that is NOT blocked.
        # Since we can't pass through the other hand, only one path is available.
        return dist_ccw if blocked_cw else dist_cw

    # Process instructions sequentially
    # We use a list to simulate the state updates since we can't use loops
    def process_instructions(idx, l, r, acc):
        if idx == Q:
            return acc
        
        h, t = instructions[idx]
        if h == 'L':
            # Move left hand to t, right hand stays at r
            dist = get_dist(l, t, r)
            return process_instructions(idx + 1, t, r, acc + dist)
        else:
            # Move right hand to t, left hand stays at l
            dist = get_dist(r, t, l)
            return process_instructions(idx + 1, l, t, acc + dist)

    # Since Q <= 100, recursion depth is fine. 
    # Using a helper function inside solve() to maintain closure.
    result = process_instructions(0, curr_l, curr_r, 0)
    print(result)

if __name__ == "__main__":
    solve()