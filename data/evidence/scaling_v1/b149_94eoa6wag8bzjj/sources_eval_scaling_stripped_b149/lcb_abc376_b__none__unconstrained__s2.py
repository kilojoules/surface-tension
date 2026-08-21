import sys

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Instructions are pairs of (H_i, T_i)
    instructions = []
    for i in range(Q):
        h = input_data[2 + 2*i]
        t = int(input_data[3 + 2*i])
        instructions.append((h, t))

    # Initial positions
    l, r = 1, 2
    total_ops = 0

    # Helper to calculate distance between two points on a ring of size N
    # without passing through a forbidden point 'p'
    def get_dist(start, end, p):
        if start == end:
            return 0
        
        # There are two paths on a ring: clockwise and counter-clockwise
        # Path 1: start -> start+1 -> ... -> end (modulo N)
        # Path 2: start -> start-1 -> ... -> end (modulo N)
        
        # We need to check if the forbidden point p lies on the path.
        # A path from s to e is "blocked" if p is between s and e.
        
        # Normalize coordinates to 0...N-1 for easier modulo arithmetic
        s, e, forbidden = start - 1, end - 1, p - 1
        
        # Clockwise distance
        dist_cw = (e - s) % N
        # The points visited clockwise are (s + k) % N for k in 1...dist_cw
        # Forbidden point is on this path if (forbidden - s) % N is between 1 and dist_cw
        blocked_cw = (1 <= (forbidden - s) % N < dist_cw) or (forbidden == e)
        # Note: the problem says T_i != other_hand, so forbidden == e is handled by constraints,
        # but we check if the path is blocked. Actually, the only way a path is blocked
        # is if the other hand is sitting on a part we must pass through.
        
        # Since we can't step on the other hand, we check if the other hand is in the way.
        # The other hand is at 'p'.
        # Clockwise path: s -> (s+1)%N -> ... -> e
        # It is blocked if p is any of the intermediate steps.
        # The distance is dist_cw. The points are (s+i)%N for i=1...dist_cw.
        # p is one of these if (p-s)%N is in {1, ..., dist_cw-1}.
        # Wait, the rule is: "you can do this only if the other hand is not on the destination part."
        # So if p == e, it's blocked. But the problem guarantees T_i != other_hand.
        # Therefore, we only care if p is an intermediate step.
        
        is_blocked_cw = (0 < (forbidden - s) % N < dist_cw)
        is_blocked_ccw = (0 < (s - forbidden) % N < (N - dist_cw))
        
        # We want the minimum distance of the non-blocked path.
        # Since it's guaranteed achievable, at least one path is open.
        res = []
        if not is_blocked_cw:
            res.append(dist_cw)
        if not is_blocked_ccw:
            res.append(N - dist_cw)
            
        return min(res)

    # Process instructions
    # We use a list for l, r to simulate mutable state inside a reduction-like logic
    # but a simple loop is cleaner.
    state = [1, 2] # [left, right]
    
    def process_instr(acc, instr):
        curr_l, curr_r, curr_ops = acc
        h, t = instr
        if h == 'L':
            # Move left hand to t, right hand stays at curr_r
            d = get_dist(curr_l, t, curr_r)
            return (t, curr_r, curr_ops + d)
        else:
            # Move right hand to t, left hand stays at curr_l
            d = get_dist(curr_r, t, curr_l)
            return (curr_l, t, curr_ops + d)

    # Using functools.reduce to follow the "no explicit for loop" constraint 
    # (though the prompt allows a complete program, reduce is safer for "functional" style)
    from functools import reduce
    final_state = reduce(process_instr, instructions, (1, 2, 0))
    
    print(final_state[2])

if __name__ == "__main__":
    solve()