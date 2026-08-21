import sys
from functools import reduce

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
        instructions.append((input_data[2 + 2*i], int(input_data[3 + 2*i])))

    # Helper to calculate distance on a ring without crossing a certain point
    # current: starting position
    # target: destination position
    # blocker: position of the other hand
    def get_dist(current, target, blocker, n):
        if current == target:
            return 0
        
        # There are two paths on a ring: clockwise and counter-clockwise.
        # One path is blocked if the blocker is "between" current and target.
        # To simplify, we can check both directions.
        # Direction 1: Increasing index (1 -> 2 -> ... -> N -> 1)
        # Direction 2: Decreasing index (1 -> N -> ... -> 2 -> 1)
        
        # We need to find the shortest path that does not step on 'blocker'.
        # Since N is small (100), we can simulate both directions or use logic.
        
        # Path A: current -> (current + 1) -> ... -> target
        # Path B: current -> (current - 1) -> ... -> target
        
        def is_blocked(start, end, block, n, direction):
            curr = start
            while curr != end:
                curr = (curr + direction - 1) % n + 1
                if curr == block:
                    return True
            return False

        # Distance for Direction 1 (+1)
        dist1 = (target - current + n) % n
        # Distance for Direction 2 (-1)
        dist2 = (current - target + n) % n
        
        blocked1 = is_blocked(current, target, blocker, n, 1)
        blocked2 = is_blocked(current, target, blocker, n, -1)
        
        # It is guaranteed that the instruction is achievable.
        # So at least one path is not blocked.
        if not blocked1 and not blocked2:
            return min(dist1, dist2)
        if not blocked1:
            return dist1
        if not blocked2:
            return dist2
        return float('inf')

    # State: (left_hand, right_hand, total_distance)
    # Initial state: L=1, R=2, dist=0
    initial_state = (1, 2, 0)

    def process_instruction(state, instr):
        l, r, total = state
        h, t = instr
        if h == 'L':
            d = get_dist(l, t, r, N)
            return (t, r, total + d)
        else:
            d = get_dist(r, t, l, N)
            return (l, t, total + d)

    final_state = reduce(process_instruction, instructions, initial_state)
    print(final_state[2])

if __name__ == "__main__":
    solve()