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

    # Helper to calculate distance between two parts on a ring of size N
    # Since we can move in both directions, the distance is the 
    # shortest path around the circle.
    # However, the problem states we cannot pass through the other hand.
    # This means we must move in the direction that doesn't encounter the other hand.
    # There are only two directions: clockwise and counter-clockwise.
    # One of them will be blocked by the other hand unless the other hand is not in the way.
    # Actually, the constraint "cannot move to the destination part if the other hand is there"
    # implies that if we want to move from A to B, and the other hand is at C,
    # we can only take the path that does not contain C.
    
    def get_dist(start, end, obstacle):
        # The two possible paths on a ring:
        # 1. Clockwise: (start + i) % N
        # 2. Counter-clockwise: (start - i) % N
        # We need the distance of the path that does not contain the obstacle.
        
        # Normalize to 0-indexed
        s, e, o = start - 1, end - 1, obstacle - 1
        
        # Path 1: s -> s+1 -> ... -> e (modulo N)
        # This path contains o if o is between s and e in clockwise direction.
        # Clockwise distance from s to e:
        dist_cw = (e - s) % N
        # Clockwise distance from s to o:
        dist_so = (o - s) % N
        
        # If dist_so < dist_cw, the obstacle is on the clockwise path.
        # Otherwise, it's on the counter-clockwise path.
        # Note: the problem guarantees the instruction is achievable.
        
        if dist_so < dist_cw:
            # Clockwise is blocked, must go counter-clockwise
            return N - dist_cw
        else:
            # Counter-clockwise is blocked or obstacle is at the end (but T_i != obstacle is guaranteed)
            # We can go clockwise.
            return dist_cw

    # We use accumulate to track the state (left_hand, right_hand, total_dist)
    # Initial state: Left=1, Right=2, Dist=0
    initial_state = (1, 2, 0)
    
    def transition(state, instr):
        l, r, d = state
        hand, target = instr
        if hand == 'L':
            # Move left hand from l to target, right hand r is the obstacle
            cost = get_dist(l, target, r)
            return (target, r, d + cost)
        else:
            # Move right hand from r to target, left hand l is the obstacle
            cost = get_dist(r, target, l)
            return (l, target, d + cost)

    # Process all instructions
    final_state = list(accumulate(instructions, transition, initial=initial_state))[-1]
    
    # The result is the total distance (the third element of the state tuple)
    print(final_state[2])

if __name__ == "__main__":
    solve()