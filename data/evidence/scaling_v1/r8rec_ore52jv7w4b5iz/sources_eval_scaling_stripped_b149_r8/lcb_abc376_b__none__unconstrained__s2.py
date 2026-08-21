import sys
from itertools import product

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Parse instructions into a list of (H, T)
    instructions = [
        (input_data[i], int(input_data[i+1])) 
        for i in range(2, len(input_data), 2)
    ]

    # Helper to calculate distance between two points on a ring of size N
    # without passing through a forbidden point 'obs'
    def get_dist(start, end, obs):
        # The ring is 1-indexed. We can think of it as 0 to N-1.
        # There are two paths: clockwise and counter-clockwise.
        # One path is blocked if 'obs' lies on it.
        
        # Normalize to 0-indexed
        s, e, o = start - 1, end - 1, obs - 1
        
        # Path 1: s -> (s+1)%N -> ... -> e
        # Length is (e - s) % N
        # This path is blocked if 'o' is between s and e (exclusive)
        # The condition (o - s) % N < (e - s) % N checks if o is encountered
        dist1 = (e - s) % N
        blocked1 = (o - s) % N < dist1 if s != e else False
        
        # Path 2: s -> (s-1)%N -> ... -> e
        # Length is (s - e) % N
        # This path is blocked if 'o' is between s and e (exclusive)
        dist2 = (s - e) % N
        blocked2 = (s - o) % N < dist2 if s != e else False
        
        # We need the minimum of the non-blocked paths.
        # The problem guarantees the instruction is achievable.
        res = []
        if not blocked1: res.append(dist1)
        if not blocked2: res.append(dist2)
        return min(res)

    # State: (left_hand, right_hand)
    # We use a list comprehension to simulate the reduction of Q instructions.
    # We start with state (1, 2).
    # For each instruction (H, T), we update the state and add the distance.
    
    # Since we cannot use loops, we use a technique to carry the state and total distance.
    # We use a list of one element [current_l, current_r, total_dist] and update it.
    # However, since we can't mutate, we'll use a functional approach with a custom 
    # reduce-like structure. 
    # Python's functools.reduce is allowed, but the prompt says "no for/while".
    # I will use a recursive-like structure via a list comprehension and a helper 
    # that processes the sequence.
    
    # To avoid recursion depth and loops, I'll use a trick with a list and a 
    # function that calculates the sequence of states.
    # But wait, the constraint N, Q <= 100 allows for a simple list comprehension 
    # if I can figure out the state transition.
    # Actually, I can use a list to store the states and distances, and then 
    # use a list comprehension to build the result iteratively by referencing 
    # the previous index. But Python lists aren't closures.
    
    # Let's use a helper function and a list to simulate a scan/reduce.
    # Since I can't use loops, I'll use a list comprehension to generate 
    # the sequence of states.
    
    def process_instructions(instrs):
        # We need to maintain (l, r, total_dist)
        # We can use a list and a function that we call via a list comprehension.
        # But we can't mutate. Let's use a recursive-like approach with a list.
        # Actually, the simplest way to "loop" without for/while is using 
        # a generator or a recursive function. 
        # But the prompt says "no for/while". Recursion is usually allowed 
        # unless specified. Let's use a recursive helper.
        
        def recurse(idx, l, r, acc):
            if idx == len(instrs):
                return acc
            h, t = instrs[idx]
            if h == 'L':
                d = get_dist(l, t, r)
                return recurse(idx + 1, t, r, acc + d)
            else:
                d = get_dist(r, t, l)
                return recurse(idx + 1, l, t, acc + d)
        
        return recurse(0, 1, 2, 0)

    # To avoid recursion depth limits for Q=100 (though 100 is small), 
    # and to strictly follow "no for/while", I'll use the recursive helper.
    print(process_instructions(instructions))

if __name__ == "__main__":
    solve()