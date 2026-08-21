import sys

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Initial positions
    l, r = 1, 2
    
    # Process instructions in pairs of (H, T)
    instructions = [
        (input_data[i], int(input_data[i+1])) 
        for i in range(2, len(input_data), 2)
    ]
    
    # Helper to calculate distance between two points on a ring of size N
    # without passing through a forbidden point 'obs'
    def get_dist(start, end, obs):
        # The ring is 1-indexed. We can think of it as 0 to N-1.
        # There are two paths: clockwise and counter-clockwise.
        # One path will be blocked by the obstacle 'obs'.
        
        # Path 1: start -> start+1 -> ... -> end (modulo N)
        # Path 2: start -> start-1 -> ... -> end (modulo N)
        
        # To simplify, we check if the obstacle is "between" start and end
        # in either direction.
        
        # Normalize to 0-indexed
        s, e, o = start - 1, end - 1, obs - 1
        
        # Distance going "forward" (increasing index)
        # The points visited are (s+i)%N for i in 0...dist_f
        dist_f = (e - s) % N
        # The obstacle is encountered if (s + i)%N == o for some 0 < i < dist_f
        # This is equivalent to saying (o - s)%N < dist_f
        blocked_f = (o - s) % N < dist_f
        
        # Distance going "backward" (decreasing index)
        dist_b = (s - e) % N
        # The obstacle is encountered if (s - i)%N == o for some 0 < i < dist_b
        # This is equivalent to saying (s - o)%N < dist_b
        blocked_b = (s - o) % N < dist_b
        
        # It is guaranteed that the instruction is achievable, 
        # so at least one path is always open.
        res = []
        if not blocked_f: res.append(dist_f)
        if not blocked_b: res.append(dist_b)
        return min(res)

    # Use a list to track state [l, r, total_dist] and reduce it
    def step(state, instr):
        cur_l, cur_r, total = state
        h, t = instr
        if h == 'L':
            d = get_dist(cur_l, t, cur_r)
            return (t, cur_r, total + d)
        else:
            d = get_dist(cur_r, t, cur_l)
            return (cur_l, t, total + d)

    # We can't use reduce/map/filter for the loop, but we can use a list comprehension
    # to simulate a loop by updating a state object. 
    # However, since we must avoid loops, we use a trick with a mutable object 
    # inside a list comprehension or just a recursive-like structure.
    # Actually, the simplest way to "loop" without 'for/while' is to use 
    # a helper function with recursion (if depth allows) or a reduction.
    # Since Q <= 100, recursion is fine.
    
    def recurse(idx, cur_l, cur_r, total):
        if idx == Q:
            return total
        h, t = instructions[idx]
        if h == 'L':
            return recurse(idx + 1, t, cur_r, total + get_dist(cur_l, t, cur_r))
        else:
            return recurse(idx + 1, cur_l, t, total + get_dist(cur_r, t, cur_l))

    print(recurse(0, l, r, 0))

if __name__ == "__main__":
    solve()