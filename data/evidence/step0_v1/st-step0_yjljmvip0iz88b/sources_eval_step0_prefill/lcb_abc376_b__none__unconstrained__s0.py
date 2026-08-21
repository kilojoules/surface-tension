import sys

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Current positions of hands
    l = 1
    r = 2
    
    total_ops = 0
    
    idx = 2
    for _ in range(Q):
        h = input_data[idx]
        t = int(input_data[idx+1])
        idx += 2
        
        if h == 'L':
            # Move left hand from l to t, while right hand r is fixed.
            # The ring is 1...N. The right hand r acts as a barrier.
            # There are two paths from l to t on the ring.
            # One path is clockwise, one is counter-clockwise.
            # However, the path cannot pass through r.
            
            # We need to find the distance from l to t avoiding r.
            # Since it's guaranteed to be achievable, there is only one path 
            # that doesn't cross r.
            
            # Let's normalize coordinates to 0...N-1 for easier modulo arithmetic
            curr_l = l - 1
            curr_r = r - 1
            target_l = t - 1
            
            # The distance is the number of steps. 
            # On a ring of size N, the distance between a and b is 
            # min(|a-b|, N-|a-b|). But we can't pass through curr_r.
            
            # To find the distance from curr_l to target_l avoiding curr_r:
            # We can simulate the movement or use the property that 
            # the only way to get from l to t is to go the "long way" 
            # if the "short way" is blocked by r.
            
            # More simply: the distance is the absolute difference if 
            # curr_r is not between curr_l and target_l (considering the ring).
            # Let's just use a simple loop to find the shortest path 
            # that doesn't hit curr_r.
            
            dist = 0
            temp_l = curr_l
            # Try moving clockwise
            # Clockwise: (x + 1) % N
            # Counter-clockwise: (x - 1) % N
            
            # Since we can only move one hand, and the other is fixed,
            # we are essentially on a line of length N-1 (the ring broken at r).
            # We can map the ring to a line by treating curr_r as the boundary.
            # Let's shift coordinates so curr_r is at N-1.
            # New coordinate x' = (x - (curr_r + 1)) % N
            # Then the available space is 0 to N-2.
            
            start = (curr_l - (curr_r + 1)) % N
            end = (target_l - (curr_r + 1)) % N
            dist = abs(start - end)
            
            total_ops += dist
            l = t
            
        else: # h == 'R'
            curr_l = l - 1
            curr_r = r - 1
            target_r = t - 1
            
            # Shift coordinates so curr_l is the boundary (at N-1)
            start = (curr_r - (curr_l + 1)) % N
            end = (target_r - (curr_l + 1)) % N
            dist = abs(start - end)
            
            total_ops += dist
            r = t
            
    print(total_ops)

if __name__ == "__main__":
    solve()