import sys

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Initial positions
    current_l = 1
    current_r = 2
    total_ops = 0
    
    # Process instructions
    # The input is a flat list: [N, Q, H1, T1, H2, T2, ...]
    # We iterate through the instructions starting from index 2
    for i in range(2, len(input_data), 2):
        h = input_data[i]
        t = int(input_data[i+1])
        
        if h == 'L':
            # We need to move Left hand from current_l to t.
            # Right hand is fixed at current_r.
            # The ring is a circle of N parts.
            # The path cannot pass through current_r.
            # There are two directions: clockwise and counter-clockwise.
            # One direction will be blocked by current_r.
            
            # Distance clockwise (increasing index)
            # To go from L to T clockwise: (T - L) % N
            # This path uses parts {L, L+1, ..., T}. 
            # It is blocked if current_r is any of these (excluding L).
            
            # Let's simplify: since we can't pass through current_r, 
            # the only available path is the one that doesn't contain current_r.
            # In a ring of N, removing one point (current_r) leaves a line of N-1 points.
            # The distance between two points on a line is simply the absolute difference
            # if we linearize the ring relative to the blocked point.
            
            # Linearization trick: 
            # Shift all indices so that current_r becomes N.
            # New index x = (x - current_r - 1) % N + 1
            # Now current_r is N, and the path between any two points 
            # cannot cross the boundary between N and 1.
            
            l_linear = (current_l - current_r - 1) % N + 1
            t_linear = (t - current_r - 1) % N + 1
            
            dist = abs(t_linear - l_linear)
            total_ops += dist
            current_l = t
            
        else: # h == 'R'
            # Same logic, but current_l is the blocked point.
            l_linear = (current_l - 1) % N + 1 # This is just current_l
            # To make current_l the "boundary", we shift:
            r_linear = (current_r - current_l - 1) % N + 1
            t_linear = (t - current_l - 1) % N + 1
            
            dist = abs(t_linear - r_linear)
            total_ops += dist
            current_r = t
            
    print(total_ops)

if __name__ == "__main__":
    solve()