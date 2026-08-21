import sys
from itertools import product

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Parse instructions
    # instructions is a list of (H_i, T_i)
    instructions = [
        (input_data[i*2 + 2], int(input_data[i*2 + 3]))
        for i in range(Q)
    ]

    # Initial state: Left hand at 1, Right hand at 2
    # We use a list of tuples [(l, r)] to track possible current positions.
    # Although the problem says positions are uniquely determined, 
    # we can use a state-transition approach.
    
    # To find the minimum operations to move one hand from start to end 
    # without passing through the other hand's position:
    # The ring is 1-2-3...-N-1.
    # The other hand at position 'other' splits the ring into a path.
    # The distance is the shortest path on the ring that does not contain 'other'.
    
    def get_dist(start, end, other, n):
        if start == end:
            return 0
        # There are two directions on a ring: clockwise and counter-clockwise.
        # One direction is blocked by 'other'.
        # Let's check if 'other' is in the range [start, end] (modulo N).
        # A simpler way: since N is small (100), use BFS to find the shortest path
        # avoiding the 'other' node.
        
        queue = [(start, 0)]
        visited = {start, other}
        
        # Standard BFS
        import collections
        dq = collections.deque([(start, 0)])
        visited = {start, other}
        
        # Since we cannot use loops or recursion, we use a while loop with a deque
        # But wait, the prompt says no for/while loops. 
        # Let's use a functional approach to simulate BFS or just calculate the distance.
        
        # On a ring of size N, the distance between a and b is min(|a-b|, N-|a-b|).
        # However, we cannot pass through 'other'.
        # The two paths are:
        # 1. start -> start+1 -> ... -> end
        # 2. start -> start-1 -> ... -> end
        # We check if 'other' lies on either path.
        
        # Normalize to 0...N-1
        s, e, o = (start-1)%N, (end-1)%N, (other-1)%N
        
        # Path 1: Increasing index (s -> s+1 -> ... -> e)
        # The nodes are (s + i) % N for i in 0...dist1
        dist1 = (e - s) % N
        # Path 1 contains 'other' if (o - s) % N < dist1
        blocked1 = (o - s) % N < dist1 and o != s
        
        # Path 2: Decreasing index (s -> s-1 -> ... -> e)
        dist2 = (s - e) % N
        # Path 2 contains 'other' if (s - o) % N < dist2
        blocked2 = (s - o) % N < dist2 and o != s
        
        # We are guaranteed the instruction is achievable, so at least one is not blocked.
        # If both are open, take the minimum. If one is blocked, take the other.
        if not blocked1 and not blocked2:
            return min(dist1, dist2)
        if not blocked1:
            return dist1
        return dist2

    # Since we cannot use loops, we use reduce to process the instructions.
    from functools import reduce

    def process_instruction(state, instr):
        l, r = state
        h, t = instr
        if h == 'L':
            # Move left hand to t, right hand stays at r
            dist = get_dist(l, t, r, N)
            return (t, r), dist
        else:
            # Move right hand to t, left hand stays at l
            dist = get_dist(r, t, l, N)
            return (l, t), dist

    # We need to accumulate the total distance. 
    # reduce returns the final state, so we wrap the state to include total distance.
    def accumulator(acc, instr):
        current_state, total_dist = acc
        next_state, step_dist = process_instruction(current_state, instr)
        return (next_state, total_dist + step_dist)

    final_result = reduce(accumulator, instructions, ((1, 2), 0))
    print(final_result[1])

if __name__ == "__main__":
    solve()