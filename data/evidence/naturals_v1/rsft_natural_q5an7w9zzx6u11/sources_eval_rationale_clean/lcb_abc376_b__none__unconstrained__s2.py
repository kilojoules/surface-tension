import sys
from functools import reduce

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Instructions as a list of (H, T)
    instructions = [
        (input_data[i], int(input_data[i+1])) 
        for i in range(2, len(input_data), 2)
    ]

    # Helper to calculate distance between a and b on a ring of size N
    # clockwise distance: (b - a) % N
    # counter-clockwise distance: (a - b) % N
    # However, we must avoid the obstacle 'obs'.
    # The only way to move from a to b is to go the direction that doesn't pass through 'obs'.
    # On a ring, there are only two paths. One is blocked if 'obs' is between a and b.
    
    def get_dist(start, end, obs):
        if start == end:
            return 0
        
        # Clockwise distance
        cw_dist = (end - start) % N
        # The clockwise path is blocked if 'obs' is encountered.
        # 'obs' is encountered if (obs - start) % N < cw_dist.
        # Wait, the rule is: you can't move to the part the other hand is holding.
        # So if we want to go clockwise from start to end, 
        # the path is start -> start+1 -> ... -> end.
        # This path is blocked if obs is any of the intermediate steps.
        
        # A simpler way:
        # There are two arcs between start and end.
        # Arc 1: start -> start+1 -> ... -> end (Length: (end-start)%N)
        # Arc 2: start -> start-1 -> ... -> end (Length: (start-end)%N)
        # One of these arcs contains 'obs'. The other doesn't.
        # Since it's guaranteed the instruction is achievable, 
        # one arc must be clear.
        
        # Check if obs is in the clockwise arc (start, end]
        # The clockwise arc consists of points (start + k) % N
        # The distance is d = (end - start) % N.
        # The points are start+1, ..., start+d.
        # obs is in this arc if (obs - start) % N is between 1 and d.
        
        cw_d = (end - start) % N
        # Adjust for 1-indexing by treating everything as 0-indexed internally
        # But since we use modulo N, as long as we are consistent, it's fine.
        # Let's use 0-indexed for logic:
        s, e, o = start - 1, end - 1, obs - 1
        
        d_cw = (e - s) % N
        # obs is in clockwise path if (o - s) % N > 0 and (o - s) % N <= d_cw
        # Actually, the only point that can block is 'o'.
        # If (o - s) % N is between 1 and d_cw, the clockwise path is blocked.
        # Note: the destination 'e' cannot be 'o' per problem statement.
        
        if 0 < (o - s) % N < d_cw:
            # Clockwise is blocked, must go counter-clockwise
            return (s - e) % N
        else:
            # Counter-clockwise is blocked or clockwise is clear
            # Check if counter-clockwise is blocked: (s - o) % N is between 1 and (s-e)%N
            # But the problem guarantees it's achievable.
            # If clockwise is not blocked, it's the only option or the shorter one?
            # No, the rule is: you cannot move the other hand.
            # You must move the specified hand to T_i.
            # You can only move through parts not occupied by the other hand.
            # This means you MUST take the arc that does not contain 'obs'.
            
            # If (o-s)%N is between 1 and d_cw, clockwise is blocked.
            # Otherwise, counter-clockwise must be blocked.
            # Let's verify: if (o-s)%N is not in (0, d_cw), then o is "outside" the cw arc.
            # That means o is in the ccw arc.
            return d_cw if (o - s) % N > d_cw or (o - s) % N == 0 else (s - e) % N

    # Corrected distance logic:
    # There are two paths from s to e.
    # Path 1 (CW): s -> s+1 -> ... -> e. Length (e-s)%N.
    # Path 2 (CCW): s -> s-1 -> ... -> e. Length (s-e)%N.
    # Path 1 is blocked if obs is any of the nodes s+1, ..., e.
    # Since e != obs, Path 1 is blocked if obs is one of s+1, ..., e-1.
    # That is, (obs-s)%N is between 1 and (e-s)%N - 1.
    
    def get_dist_final(s, e, o):
        if s == e: return 0
        s, e, o = s-1, e-1, o-1
        d_cw = (e - s) % N
        # Path 1 (CW) is blocked if o is in {s+1, ..., e-1}
        # This happens if 0 < (o - s) % N < d_cw
        if 0 < (o - s) % N < d_cw:
            return (s - e) % N # Must go CCW
        else:
            return d_cw # Must go CW

    # State: (current_L, current_R, total_dist)
    def step(state, instr):
        l, r, dist = state
        h, t = instr
        if h == 'L':
            return (t, r, dist + get_dist_final(l, t, r))
        else:
            return (l, t, dist + get_dist_final(r, t, l))

    final_state = reduce(step, instructions, (1, 2, 0))
    print(final_state[2])

if __name__ == "__main__":
    solve()