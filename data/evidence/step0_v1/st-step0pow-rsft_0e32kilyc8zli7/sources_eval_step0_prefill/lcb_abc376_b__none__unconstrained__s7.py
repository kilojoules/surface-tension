import sys

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Initial positions
    l = 1
    r = 2
    total_ops = 0
    
    idx = 2
    for _ in range(Q):
        h = input_data[idx]
        t = int(input_data[idx+1])
        idx += 2
        
        if h == 'L':
            # Move left hand from l to t, right hand r is fixed.
            # The ring is 1...N. The right hand r acts as a barrier.
            # There are two paths from l to t: clockwise and counter-clockwise.
            # One path is blocked by r.
            
            # We need to find the distance from l to t avoiding r.
            # Let's represent positions on a circle.
            # The available path is the arc from l to t that does not contain r.
            
            # To simplify, we can think of the ring as a linear array by 
            # "cutting" it at r. However, it's easier to just check 
            # which direction is blocked.
            
            # Distance moving "up" (1 -> 2 -> ... -> N -> 1)
            # Distance moving "down" (1 -> N -> ... -> 2 -> 1)
            
            # The only way to move from l to t is to go the long way around 
            # if the short way is blocked by r.
            # Actually, since r is fixed, there is only one contiguous 
            # segment of the ring available: the one from r+1 to r-1 (mod N).
            
            # Let's normalize coordinates to 0...N-1
            curr_l = l - 1
            curr_r = r - 1
            target_l = t - 1
            
            # The available nodes are those in the range (curr_r, curr_r) 
            # excluding curr_r itself.
            # The distance is simply the distance from curr_l to target_l 
            # in the direction that doesn't pass through curr_r.
            
            # Calculate clockwise distance from l to t
            dist_cw = (target_l - curr_l + N) % N
            # Check if r is in the clockwise path from l to t
            # r is in path if (curr_r - curr_l + N) % N < dist_cw
            if (curr_r - curr_l + N) % N < dist_cw:
                # Clockwise is blocked, must go counter-clockwise
                dist = (curr_l - target_l + N) % N
            else:
                # Clockwise is clear
                dist = dist_cw
            
            total_ops += dist
            l = t
            
        else: # h == 'R'
            curr_l = l - 1
            curr_r = r - 1
            target_r = t - 1
            
            dist_cw = (target_r - curr_r + N) % N
            if (curr_l - curr_r + N) % N < dist_cw:
                dist = (curr_r - target_r + N) % N
            else:
                dist = dist_cw
                
            total_ops += dist
            r = t
            
    print(total_ops)

if __name__ == "__main__":
    solve()