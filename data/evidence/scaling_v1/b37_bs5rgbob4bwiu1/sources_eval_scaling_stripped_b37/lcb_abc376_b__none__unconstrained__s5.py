import sys
from itertools import product

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    instructions = []
    idx = 2
    for _ in range(Q):
        h = input_data[idx]
        t = int(input_data[idx+1])
        instructions.append((h, t))
        idx += 2

    # State: (left_hand, right_hand)
    # Initial state: (1, 2)
    # We want to find the minimum operations to satisfy Q instructions.
    # Since N and Q are small (100), and the other hand must remain stationary,
    # the movement of the active hand is constrained to the arc not containing the other hand.
    
    # Function to calculate shortest distance between a and b on a ring of size N
    # without passing through point c.
    # The ring is 1-indexed.
    def get_dist(start, end, obstacle):
        # We can move clockwise or counter-clockwise.
        # One direction is blocked by the obstacle.
        # Let's check both directions.
        
        # Clockwise distance (increasing index)
        # To go from start to end clockwise:
        # If we don't cross the obstacle, the distance is (end - start) % N
        # But we must check if the obstacle lies on the path.
        
        # A simpler way: 
        # There are only two paths between any two points on a ring.
        # Path 1: start -> start+1 -> ... -> end
        # Path 2: start -> start-1 -> ... -> end
        
        # We check if the obstacle is "inside" the range.
        # Since we can't jump over the obstacle, we check if the path is clear.
        
        # Distance clockwise
        cw_dist = (end - start) % N
        # The nodes visited clockwise are (start + i) % N
        # The obstacle is at 'obstacle'. 
        # The obstacle is hit if (obstacle - start) % N < cw_dist
        # Wait, the above is for 0-indexed. Let's use 0-indexed internally.
        return None

    # Let's redefine using 0-indexing
    # start, end, obs are 0...N-1
    def dist_without(s, e, obs):
        if s == e:
            return 0
        # Clockwise distance
        cw_dist = (e - s) % N
        # Obstacle is hit if (obs - s) % N < cw_dist
        # Actually, the only way to get from s to e is clockwise or counter-clockwise.
        # If the obstacle is in the way clockwise, we MUST go counter-clockwise.
        # If it's in the way counter-clockwise, we MUST go clockwise.
        # It is guaranteed that the instruction is achievable, so one path is always clear.
        
        # Check if clockwise path is clear:
        # The clockwise path consists of nodes { (s + i) % N for i in 1..cw_dist }
        # The obstacle is hit if (obs - s) % N is between 1 and cw_dist-1.
        # Wait, the rule is "destination part". So we can't move TO the obstacle.
        # So if (obs - s) % N < cw_dist, the clockwise path is blocked.
        
        if (obs - s) % N < cw_dist:
            # Clockwise is blocked, must go counter-clockwise
            return (s - e) % N
        else:
            # Clockwise is clear
            return cw_dist

    # Initial state
    curr_l, curr_r = 0, 1 # 0-indexed (1, 2)
    total_ops = 0
    
    for h, t in instructions:
        target = t - 1 # 0-indexed
        if h == 'L':
            total_ops += dist_without(curr_l, target, curr_r)
            curr_l = target
        else:
            total_ops += dist_without(curr_r, target, curr_l)
            curr_r = target
            
    print(total_ops)

if __name__ == "__main__":
    solve()