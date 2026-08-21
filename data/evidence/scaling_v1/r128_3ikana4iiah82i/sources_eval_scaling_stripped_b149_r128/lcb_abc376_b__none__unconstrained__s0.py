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
    # distance(a, b) = min(|a-b|, N - |a-b|)
    def get_dist(a, b):
        diff = abs(a - b)
        return min(diff, N - diff)

    # We need to track the positions of both hands (L, R)
    # Initial state: L=1, R=2
    # For each instruction (H, T):
    # If H == 'L', L moves to T, R stays.
    # If H == 'R', R moves to T, L stays.
    # The cost is the distance the moving hand travels.
    # However, the moving hand cannot pass through the stationary hand.
    # On a ring, there are two paths. One is blocked by the other hand.
    # The only available path is the one that doesn't contain the other hand's position.
    
    # To determine the distance without passing the other hand:
    # If we move from 'start' to 'end' and 'obstacle' is in the way:
    # The distance is simply the length of the path that avoids the obstacle.
    # Since it's guaranteed the instruction is achievable, the obstacle 
    # must not be at the destination.
    
    # The distance between start and end is d. 
    # One path has length d, the other N-d.
    # The obstacle is at some position. 
    # If the obstacle is "between" start and end on the short path, 
    # we must take the long path.
    
    # A more robust way to check if the obstacle is on the path:
    # For a move from s to e, the path is "clockwise" or "counter-clockwise".
    # We check which direction is blocked by the obstacle.
    
    def calc_move_dist(s, e, obs):
        if s == e:
            return 0
        # Distance if moving clockwise (increasing index)
        # From s to e clockwise:
        dist_cw = (e - s + N) % N
        # The obstacle is on the clockwise path if (obs - s + N) % N < dist_cw
        # But the obstacle is the other hand. 
        # We can move clockwise if the obstacle is not in the range (s, e) clockwise.
        # The obstacle is at 'obs'.
        # It is on the clockwise path if (obs - s + N) % N < dist_cw
        # Note: the problem says we can't move to the destination if the other hand is there.
        # It also implies we can't jump over the other hand.
        
        # Check if obstacle is on the clockwise path
        # The clockwise path consists of nodes: (s+1)%N, (s+2)%N ... e
        # The obstacle is on this path if (obs - s + N) % N < dist_cw
        # Wait, the obstacle could be at s, but the rules say we move TO an adjacent part.
        # The obstacle is the other hand.
        
        # If the obstacle is on the clockwise path, we must go counter-clockwise.
        # If the obstacle is on the counter-clockwise path, we must go clockwise.
        # Since it's guaranteed to be achievable, the obstacle cannot be on both paths
        # (which is impossible anyway since N >= 3) and cannot be at the destination.
        
        # Clockwise distance
        d_cw = (e - s + N) % N
        # Is obstacle on the clockwise path?
        # The nodes are s+1, s+2 ... e. 
        # The obstacle is on the path if (obs - s + N) % N is between 1 and d_cw-1.
        # Actually, the only way to get from s to e is to NOT pass through obs.
        # There are only two paths. One is blocked if obs is on it.
        
        # Let's use a simpler logic:
        # The distance is either (e-s)%N or (s-e)%N.
        # One of these paths contains 'obs'.
        # The path from s to e clockwise is blocked if (obs-s)%N < (e-s)%N.
        # But the obstacle could be at s. The rule says "other hand is not on the destination".
        # It doesn't say the obstacle can't be at the start.
        # However, the hands start at 1 and 2. They will never be on the same part.
        
        # Correct logic for ring distance avoiding an obstacle:
        # The two paths are:
        # 1. s -> s+1 -> ... -> e (Clockwise)
        # 2. s -> s-1 -> ... -> e (Counter-clockwise)
        # Path 1 is blocked if obs is any of the parts between s and e.
        # Path 2 is blocked if obs is any of the parts between s and e.
        
        # Since we can't pass the obstacle, we must take the path that doesn't contain it.
        # The distance is N - (distance of the blocked path).
        # But we don't know which one is blocked.
        # Actually, the only path available is the one where obs is NOT encountered.
        # If we move clockwise, we encounter obs if (obs - s + N) % N < (e - s + N) % N.
        # BUT, the obstacle is the other hand. The other hand is at 'obs'.
        # We can move clockwise if for all k from 1 to dist_cw, (s + k) % N != obs.
        # This is equivalent to saying (obs - s + N) % N > dist_cw or (obs - s + N) % N == 0.
        # Wait, the obstacle is the other hand. It's always at some position.
        # The distance is simply the distance from s to e in the direction that doesn't hit obs.
        
        # Let's use the property: the only available path is the one that doesn't contain 'obs'.
        # The distance is then the length of that path.
        # If the clockwise path is blocked, the distance is the counter-clockwise distance.
        # Otherwise, it's the clockwise distance.
        
        # Is the clockwise path blocked?
        # It's blocked if (obs - s + N) % N < (e - s + N) % N.
        # Note: (obs - s + N) % N is the clockwise distance from s to obs.
        # If this is less than the clockwise distance from s to e, then obs is on the way.
        
        dist_cw = (e - s + N) % N
        obs_dist_cw = (obs - s + N) % N
        
        if obs_dist_cw < dist_cw:
            # Clockwise is blocked, must go counter-clockwise
            return (s - e + N) % N
        else:
            # Clockwise is not blocked (obs is either at s or beyond e)
            # But wait, if obs is at s, it doesn't block the move to s+1.
            # The only way the clockwise path is blocked is if the obstacle is 
            # at some position strictly between s and e.
            # However, the problem says we can't move to the destination if the other hand is there.
            # It doesn't say we can't move if the other hand is at the start.
            # But the hands are always at different positions.
            # So obs != s and obs != e.
            # Therefore, the clockwise path is blocked if 0 < obs_dist_cw < dist_cw.
            # Since obs != s, obs_dist_cw is always > 0.
            # So the condition is simply obs_dist_cw < dist_cw.
            return dist_cw

    # Use accumulate to simulate the process and track (current_L, current_R, total_dist)
    # Initial state: L=1, R=2, dist=0
    final_state = list(accumulate(
        instructions,
        lambda state, inst: (
            (state[0], inst[1], state[2] + calc_move_dist(state[1], inst[1], state[0]))
            if inst[0] == 'R' else
            (inst[1], state[1], state[2] + calc_move_dist(state[0], inst[1], state[1]))
        ),
        initial=(1, 2, 0)
    ))
    
    # The result is the total distance of the last state
    print(final_state[-1][2])

if __name__ == "__main__":
    solve()