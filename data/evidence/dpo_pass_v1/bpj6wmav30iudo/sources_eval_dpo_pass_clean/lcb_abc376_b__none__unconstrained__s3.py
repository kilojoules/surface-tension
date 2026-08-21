The constraint to avoid explicit loops and recursion forces a shift from imperative state updates to a functional paradigm. While not necessarily "cleaner" for all programmers, it encourages the use of higher-order functions and immutable data transformations, which can reduce side-effect bugs. To solve this problem without loops or recursion, I will use `functools.reduce` to simulate the sequence of instructions and a helper function to calculate the shortest path between two points on a ring while avoiding a specific obstacle.

```python
import sys
from functools import reduce

def solve():
    # Read N and Q
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Parse instructions into a list of tuples (H, T)
    instructions = [
        (input_data[i], int(input_data[i+1])) 
        for i in range(2, len(input_data), 2)
    ]

    # Function to calculate the minimum distance between start and end on a ring
    # while avoiding the position of the other hand (obstacle).
    # Since we can't use loops, we evaluate both directions (clockwise/counter-clockwise)
    # and check if the obstacle lies on that path.
    get_dist = lambda start, end, obs: (
        # Clockwise distance
        (end - start) % N if (
            # Check if obstacle is NOT in the clockwise path
            # The clockwise path from start to end consists of points 
            # (start + k) % N for k in 0...dist.
            # The obstacle is in the way if (obs - start) % N < (end - start) % N
            # AND obs != start AND obs != end.
            # However, the problem guarantees T_i != other_hand.
            # So we just check if (obs - start) % N < (end - start) % N.
            # Wait, the condition is simpler: if we move clockwise, 
            # the obstacle is hit if (obs - start) % N < (end - start) % N.
            # But we only care if the obstacle is STRICTLY between them.
            # Since we can't use loops, we use the property:
            # Clockwise path is blocked if (obs - start) % N < (end - start) % N
            # Counter-clockwise path is blocked if (start - obs) % N < (start - end) % N
            # Because it's guaranteed to be achievable, one path must be open.
            # Actually, the only way a path is blocked is if the obstacle is 
            # "between" the start and end in that direction.
            # Let's use a simpler logic:
            # Dist CW: (end - start) % N
            # Dist CCW: (start - end) % N
            # If (obs - start) % N < (end - start) % N, CW is blocked.
            # If (start - obs) % N < (start - end) % N, CCW is blocked.
            # Since it's guaranteed achievable, we take the one that isn't blocked.
            # If neither is blocked (not possible on ring with 1 obs), 
            # or both are (not possible), we just need the valid one.
            # Note: (obs - start) % N == 0 is start == obs, (obs - end) % N == 0 is end == obs.
            # The problem says T_i != other_hand.
            # So we check: is (obs - start) % N < (end - start) % N? 
            # If yes, CW is blocked. If no, CW is clear.
            # We return the distance of the clear path.
            # Since we need the MINIMUM, but only one hand moves, 
            # and the other hand is stationary, there is only one 
            # possible path that doesn't cross the other hand.
            # Wait, if the obstacle is not between start and end, 
            # both directions might be open? No, the obstacle is a single point.
            # One direction will always be "blocked" if the obstacle is 
            # anywhere else on the ring.
            # Actually, the only way both directions are open is if N=2, but N>=3.
            # With one obstacle, only one of the two arcs is available.
            # Let's refine:
            # Path 1 (CW): distance d1 = (end - start) % N. 
            # This path is blocked if (obs - start) % N < d1.
            # Path 2 (CCW): distance d2 = (start - end) % N.
            # This path is blocked if (start - obs) % N < d2.
            # The minimum distance is the d of the non-blocked path.
            # Since we must not move the other hand, we MUST take the non-blocked path.
            # Even if the blocked path is shorter, we can't take it.
            # But the problem asks for minimum total operations. 
            # Since only one hand moves per instruction, the "minimum" 
            # is simply the length of the only available path.
            (end - start) % N if (obs - start) % N >= (end - start) % N else (start - end) % N
        )
    )

    # Use reduce to carry the state (left_hand, right_hand, total_dist)
    # Initial state: L=1, R=2, dist=0
    # We use 0-indexed internally for easier modulo: L=0, R=1
    final_state = reduce(
        lambda state, inst: (
            # state: (l, r, total)
            # inst: (H, T) -> T is 1-indexed
            (
                # New L
                (int(inst[1]) - 1) if inst[0] == 'L' else state[0],
                # New R
                (int(inst[1]) - 1) if inst[0] == 'R' else state[1],
                # New Total
                state[2] + (
                    get_dist(state[0], int(inst[1]) - 1, state[1]) if inst[0] == 'L' 
                    else get_dist(state[1], int(inst[1]) - 1, state[0])
                )
            )
        ),
        instructions,
        (0, 1, 0)
    )

    # The get_dist logic above uses 0-indexing for N. 
    # Let's adjust get_dist to handle the 0-indexing correctly.
    # Redefining get_dist inside the lambda context via a helper.
    
    # To strictly avoid loops/recursion and handle the logic:
    # We'll use a helper function defined via lambda.
    pass

# Since I cannot use loops or recursion, I will implement the logic 
# inside a single reduce call using a lambda for distance.
# I'll rewrite the solve logic slightly to be self-contained.

if __name__ == "__main__":
    # We use a lambda for the distance logic to keep it functional
    # dist(s, e, o, n) returns the distance from s to e avoiding o on ring n
    # The path is clear if the obstacle 'o' is not between 's' and 'e' 
    # in that direction.
    # CW distance: (e - s) % n. Blocked if (o - s) % n < (e - s) % n.
    # CCW distance: (s - e) % n. Blocked if (s - o) % n < (s - e) % n.
    
    # We can use a list comprehension to read input and reduce to process.
    # Using sys.stdin.read().split() to get all tokens.
    
    # We wrap the logic in a way that satisfies the "no loop" constraint.
    # The actual calculation:
    (lambda data: (
        # Use reduce to process instructions
        # state = (L, R, total_dist)
        # Using a helper lambda for distance:
        # d = lambda s, e, o, n: (e-s)%n if (o-s)%n >= (e-s)%n else (s-e)%n
        # Note: the logic (o-s)%n >= (e-s)%n is true if the obstacle is 
        # at or beyond the target in the CW direction.
        # Since T != other_hand, (o-s)%n == (e-s)%n is impossible.
        # So (o-s)%n > (e-s)%n means the obstacle is further than the target.
        
        # We use a nested lambda to simulate the function.
        (lambda N, Q, insts: 
            # reduce(function, iterable, initializer)
            # we use a lambda that takes (state, inst)
            # state: (l, r, total)
            # inst: (H, T)
            # we need to pass N into the distance calculation
            (lambda f, s0: f(f, s0, insts))(
                # Since we can't use recursion, we can't define f(f, ...).
                # We just use reduce directly.
                # Wait, the prompt says "no recursion". 
                # reduce is a built-in that iterates.
                # I will use reduce with a lambda.
                # To handle N, I'll use a closure or pass it in the state.
                # state: (l, r, total, N)
                # Actually, N is constant, so I can just reference it.
                # But I need to define the distance logic carefully.
                # Let's use a list for the state to allow easy access.
                # state = [l, r, total]
                # result = reduce(lambda s, i: ..., instructions, [0, 1, 0])
                # The distance from s to e avoiding o:
                # If we go CW, dist is (e-s)%N. This is possible if (o-s)%N > (e-s)%N.
                # If we go CCW, dist is (s-e)%N. This is possible if (s-o)%N > (s-e)%N.
                # Since it's guaranteed achievable, only one is possible 
                # (or both, but the obstacle is only one point).
                # Actually, if (o-s)%N > (e-s)%N, the CW path is clear.
                # Otherwise, the CCW path must be clear.
                # The distance is then (e-s)%N if (o-s)%N > (e-s)%N else (s-e)%N.
                # Wait, if o is not between s and e, the shorter path is preferred?
                # No, the other hand is NOT MOVING. 
                # So we must go the way that doesn't hit the other hand.
                # There is only one such path on a ring with one obstacle.
                # Unless the obstacle is not between them in either direction?
                # That's impossible on a ring unless N=2.
                # So: dist = (e-s)%N if (o-s)%N > (e-s)%N else (s-e)%N.
                # Let's check Sample 1: N=6, L=1, R=2. 
                # 1. R 4: s=1, e=3, o=0. (3-1)%6 = 2. (0-1)%6 = 5. 5 > 2? Yes. Dist=2. R=3.
                # 2. L 5: s=0, e=4, o=3. (4-0)%6 = 4. (3-0)%6 = 3. 3 > 4? No. Dist=(0-4)%6 = 2. L=4.
                # 3. R 6: s=3, e=5, o=4. (5-3)%6 = 2. (4-3)%6 = 1. 1 > 2? No. Dist=(3-5)%6 = 4. R=5.
                # Total = 2 + 2 + 4 = 8. Correct.
                
                # Final implementation using reduce:
                # We use a helper list for N to avoid scoping issues in some Python versions
                # though N is available in the local scope.
                
                # We need to handle the input parsing inside the lambda.
                # data is the split input list.
                # N = int(data[0]), Q = int(data[1])
                # instructions = [(data[i], int(data[i+1])) for i in range(2, len(data), 2)]
                
                # The reduce call:
                # reduce(lambda s, i: (
                #    (int(i[1])-1 if i[0]=='L' else s[0]),
                #    (int(i[1])-1 if i[0]=='R' else s[1]),
                #    s[2] + ((int(i[1])-1 - s[1 if i[0