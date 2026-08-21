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
    # instructions = [('R', 4), ('L', 5), ('R', 6)]
    instructions = [
        (input_data[i*2 + 2], int(input_data[i*2 + 3])) 
        for i in range(Q)
    ]

    # Helper to calculate shortest distance between two parts on a ring of size N
    # The distance is the number of edges between them.
    def get_dist(a, b):
        return min(abs(a - b), N - abs(a - b))

    # We need to track the state (left_hand, right_hand, total_cost)
    # Initial state: Left hand at 1, Right hand at 2, Cost 0
    initial_state = (1, 2, 0)

    # The core logic of the movement:
    # To move hand H to target T without moving the other hand O:
    # The other hand O acts as a barrier. 
    # The ring is split into two paths between current H and T.
    # One path is 'clockwise', one is 'counter-clockwise'.
    # If the barrier O is on a path, that path is blocked.
    # However, the problem guarantees the instruction is achievable.
    # A path is blocked if the barrier O is strictly between H and T 
    # on that specific arc of the ring.
    
    def calculate_step(state, instr):
        l, r, cost = state
        hand, target = instr
        
        if hand == 'L':
            # Moving left hand from l to target, right hand r is fixed.
            # We need the distance from l to target avoiding r.
            # The two possible distances are:
            # 1. Direct distance (if r is not in the way)
            # 2. The long way around (if r is not in the way)
            # Actually, since we can't pass through r, there is only one valid path.
            # The distance is the length of the arc from l to target that does not contain r.
            
            # To find the distance from l to target avoiding r:
            # We can simulate the two directions.
            # Direction 1: l -> l+1 -> ... -> target (mod N)
            # Direction 2: l -> l-1 -> ... -> target (mod N)
            
            # Instead of loops, we use the property:
            # The total distance is N. The distance from l to r is d1, r to target is d2.
            # The path that doesn't contain r is the one that doesn't "cross" r.
            # A simpler way: the distance is the absolute difference if r is not between them.
            # But the ring makes it tricky. 
            # Let's use the property: distance is min(abs(l-target), N-abs(l-target))
            # UNLESS the barrier r is on that shortest path.
            # If the barrier is on the shortest path, we MUST take the longer path.
            
            # Check if r is on the shortest path between l and target:
            # r is on the shortest path if dist(l, r) + dist(r, target) == dist(l, target)
            d_lt = get_dist(l, target)
            if get_dist(l, r) + get_dist(r, target) == d_lt:
                # Shortest path is blocked, must take the other way.
                # The other way is N - d_lt.
                # Note: if N=3 and l=1, target=2, r=3, the shortest path is 1, but 
                # the barrier is at 3. The barrier is NOT on the path 1->2.
                # Wait, if dist(l, r) + dist(r, target) == dist(l, target), 
                # it means r is "between" l and target on the shortest arc.
                move_cost = N - d_lt
            else:
                move_cost = d_lt
            
            return (target, r, cost + move_cost)
        
        else: # hand == 'R'
            # Moving right hand from r to target, left hand l is fixed.
            d_rt = get_dist(r, target)
            if get_dist(r, l) + get_dist(l, target) == d_rt:
                move_cost = N - d_rt
            else:
                move_cost = d_rt
                
            return (l, target, cost + move_cost)

    # Use accumulate to process all instructions and find the final state
    final_state = list(accumulate(instructions, calculate_step, initial=initial_state))[-1]
    
    # The result is the total cost (the third element of the state tuple)
    print(final_state[2])

if __name__ == "__main__":
    solve()