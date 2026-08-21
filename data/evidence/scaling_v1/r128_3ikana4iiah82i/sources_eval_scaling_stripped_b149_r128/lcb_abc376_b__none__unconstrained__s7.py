import sys
from itertools import accumulate

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Parse instructions into a list of (hand, target)
    # H_i is at index 2 + 2*i, T_i is at index 3 + 2*i
    instructions = [
        (input_data[2 + 2*i], int(input_data[3 + 2*i]))
        for i in range(Q)
    ]

    # Helper to calculate shortest distance between two points on a ring of size N
    # without passing through a forbidden point.
    # Since we can only move one hand, the other hand acts as a wall.
    # The distance is simply the distance in the direction that doesn't hit the wall.
    def get_dist(start, end, wall):
        if start == end:
            return 0
        
        # We need to check if the wall is "between" start and end in either direction.
        # Let's normalize everything relative to start = 0.
        # The ring is 0 to N-1.
        s = 0
        e = (end - start) % N
        w = (wall - start) % N
        
        # Clockwise distance is e. 
        # The wall is in the way if 0 < w < e.
        # Counter-clockwise distance is N - e.
        # The wall is in the way if e < w < N.
        
        # It is guaranteed that the instruction is achievable.
        # If the wall is not in the clockwise path, we can take it.
        # Otherwise, we must take the counter-clockwise path.
        # Note: the problem says we cannot move to the destination if the other hand is there.
        # But the guarantee says T_i != other_hand.
        
        # Check if wall is in the path 0 -> 1 -> ... -> e
        # The wall is at position w.
        if w == 0 or w == e:
            # This case is technically excluded by the problem guarantee (T_i != wall)
            # and the fact that the hand being moved is the one at 'start'.
            # However, if the wall is at the start, it doesn't block the first move.
            # If the wall is at the end, it's forbidden by the guarantee.
            pass

        # We can move clockwise if there is no wall at 1, 2, ..., e-1
        # Actually, the only way the wall blocks the path is if it's strictly between.
        # But since we can't step ON the wall, any w in {1, ..., e-1} blocks.
        # Wait, the rule is: "you can do this only if the other hand is not on the destination part."
        # So if the wall is at position w, we cannot move to w.
        
        # Clockwise path: start -> start+1 -> ... -> end
        # This path is blocked if the wall is any of the intermediate steps.
        # The intermediate steps are (start + k) % N for k = 1 to dist_cw - 1.
        # But the rule is we can't move to the destination if the wall is there.
        # The guarantee says T_i != wall, so the destination is safe.
        # The only thing that can stop us is if the wall is an intermediate step.
        
        # Let's use a simpler logic:
        # There are two paths: 
        # 1. Clockwise: distance is (end - start) % N
        # 2. Counter-clockwise: distance is (start - end) % N
        # A path is blocked if the wall is encountered.
        
        dist_cw = (end - start) % N
        dist_ccw = (start - end) % N
        
        # Wall position relative to start
        w_rel = (wall - start) % N
        
        # Wall blocks clockwise if 0 < w_rel < dist_cw
        # Wall blocks counter-clockwise if dist_cw < w_rel < N
        
        # Since it's guaranteed to be achievable, one of these must be false.
        # We want the minimum distance of the non-blocked path.
        
        # If both are open (only possible if N is large and wall is far), take min.
        # But the wall is always somewhere.
        
        # Let's refine:
        # Path CW is blocked if w_rel is in {1, 2, ..., dist_cw - 1}
        # Path CCW is blocked if w_rel is in {dist_cw + 1, ..., N - 1}
        
        # Note: if dist_cw == 0, distance is 0.
        # If dist_cw == 1, the CW path is just one step. No intermediate steps.
        # The only way it's blocked is if the destination is the wall, but that's forbidden.
        
        # So:
        can_cw = not (0 < w_rel < dist_cw)
        can_ccw = not (dist_cw < w_rel < N)
        
        # We return the distance of the path that is not blocked.
        # If both are not blocked, we take the minimum.
        # (Though with one wall, usually only one path is open unless the wall is at the start/end)
        
        res = [dist_cw if can_cw else None, dist_ccw if can_ccw else None]
        return min(filter(lambda x: x is not None, res))

    # Initial state
    initial_state = {'L': 1, 'R': 2, 'total_dist': 0}

    # Use accumulate to process instructions and track the state (L, R, total_dist)
    def step(state, instr):
        hand, target = instr
        l, r, total = state
        if hand == 'L':
            d = get_dist(l, target, r)
            return (target, r, total + d)
        else:
            d = get_dist(r, target, l)
            return (l, target, total + d)

    # We use a list/tuple for state in accumulate: (left_hand, right_hand, total_distance)
    final_state = list(accumulate(instructions, step, initial=(1, 2, 0)))
    
    # The result is the total_dist of the last state
    print(final_state[-1][2])

if __name__ == "__main__":
    solve()