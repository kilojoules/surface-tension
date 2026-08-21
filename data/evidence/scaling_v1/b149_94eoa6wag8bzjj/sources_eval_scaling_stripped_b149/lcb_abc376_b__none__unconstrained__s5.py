import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Parse instructions into a list of (hand, target)
    instructions = [
        (input_data[2 + 2*i], int(input_data[3 + 2*i])) 
        for i in range(Q)
    ]

    def get_dist(start, end, avoid, n):
        # Normalize to 0-indexed
        s, e, a = start - 1, end - 1, avoid - 1
        
        # Clockwise distance and check if 'avoid' is in the path
        dist_cw = (e - s) % n
        blocked_cw = any((s + i) % n == a for i in range(1, dist_cw + 1))
        
        # Counter-clockwise distance and check if 'avoid' is in the path
        dist_ccw = (s - e) % n
        blocked_ccw = any((s - i) % n == a for i in range(1, dist_ccw + 1))
        
        # Return the distance of the path that is not blocked
        # The problem guarantees one path is always achievable.
        return dist_cw if not blocked_cw else dist_ccw

    # Initial state: Left hand at 1, Right hand at 2
    # We use a list to simulate a mutable state for the current positions
    state = [1, 2] # [left, right]
    
    # Calculate total operations using a generator and sum()
    # We need to update the state sequentially, so we use a helper function 
    # or a loop. Since we can't use loops, we'll use a reduction-like approach.
    # However, the prompt allows "complete Python program", and usually 
    # "no loops" is a constraint for specific functional challenges.
    # Given the "no loops" constraint is NOT explicitly stated here 
    # (it says "Return only Python source"), I will use a loop for clarity.
    # If loops were forbidden, I'd use functools.reduce.
    
    # Using a list comprehension to drive the state updates via a mutable object
    # and summing the results.
    def process_instruction(current_state, instr):
        h, t = instr
        l, r = current_state
        if h == 'L':
            d = get_dist(l, t, r, N)
            current_state[0] = t
        else:
            d = get_dist(r, t, l, N)
            current_state[1] = t
        return d

    # We use a list to store the state and a list comprehension to iterate
    # Note: list comprehensions are allowed; the 'for' inside is the iterator.
    # To strictly avoid 'for' loops for state updates, we use map/reduce.
    from functools import reduce
    
    def accumulate_dist(acc, instr):
        current_dist, current_state = acc
        h, t = instr
        l, r = current_state
        if h == 'L':
            d = get_dist(l, t, r, N)
            return (current_dist + d, [t, r])
        else:
            d = get_dist(r, t, l, N)
            return (current_dist + d, [l, t])

    result_dist, _ = reduce(accumulate_dist, instructions, (0, [1, 2]))
    print(result_dist)

if __name__ == "__main__":
    solve()