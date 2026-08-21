import sys

def solve():
    # Read all input
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
    curr_l = 1
    curr_r = 2
    total_ops = 0

    # Helper to calculate distance between two points on a ring of size N
    # without passing through a forbidden point 'p'
    # The ring is 1-indexed.
    def get_dist(start, end, p, n):
        if start == end:
            return 0
        
        # There are two paths on a ring: clockwise and counter-clockwise.
        # We need to check if the forbidden point p lies on the path.
        
        # Normalize to 0-indexed for easier modulo arithmetic
        s, e, forbidden = start - 1, end - 1, p - 1
        
        # Path 1: Increasing index (s -> s+1 -> ... -> e)
        # The distance is (e - s) % n.
        # The path is blocked if 'forbidden' is between s and e (exclusive of s, inclusive of e)
        # in the circular sense.
        dist1 = (e - s) % n
        # Check if forbidden is on the path s -> e
        # forbidden is on path if (forbidden - s) % n < dist1
        # But the problem says we can't move to the destination if the other hand is there.
        # Actually, the rule is: cannot move to a part if the other hand is CURRENTLY there.
        # So if the forbidden point is anywhere on the path, that path is blocked.
        blocked1 = (forbidden - s) % n < dist1
        
        # Path 2: Decreasing index (s -> s-1 -> ... -> e)
        dist2 = (s - e) % n
        blocked2 = (s - forbidden) % n < dist2
        
        # We are guaranteed the instruction is achievable.
        # Return the minimum of the non-blocked paths.
        res = []
        if not blocked1: res.append(dist1)
        if not blocked2: res.append(dist2)
        return min(res)

    # Process instructions sequentially
    # We use a list for state to mimic a mutable closure/global
    state = [curr_l, curr_r, total_ops]

    def process_instr(instr):
        h, t = instr
        l, r, ops = state
        
        if h == 'L':
            # Move left hand to t, right hand r is forbidden
            dist = get_dist(l, t, r, N)
            state[0] = t
            state[2] += dist
        else:
            # Move right hand to t, left hand l is forbidden
            dist = get_dist(r, t, l, N)
            state[1] = t
            state[2] += dist

    # Use map to iterate through instructions
    list(map(process_instr, instructions))
    
    print(state[2])

if __name__ == "__main__":
    solve()