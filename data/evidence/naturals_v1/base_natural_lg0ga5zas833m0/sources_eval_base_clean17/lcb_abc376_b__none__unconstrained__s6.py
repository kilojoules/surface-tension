import sys

def solve():
    # Read N and Q
    try:
        line1 = sys.stdin.readline().split()
        if not line1:
            return
        n, q = map(int, line1)
    except ValueError:
        return

    # Current positions of hands
    l, r = 1, 2
    total_ops = 0

    for _ in range(q):
        line = sys.stdin.readline().split()
        if not line:
            break
        h = line[0]
        t = int(line[1])

        if h == 'L':
            # Move left hand from l to t, right hand r stays fixed
            # The left hand can move in two directions: clockwise or counter-clockwise.
            # However, it cannot pass through the position of the right hand r.
            # The ring is 1, 2, ..., N.
            
            # Calculate distance moving clockwise and counter-clockwise
            # Distance is the number of edges.
            # Since we can't cross r, we check if r is "in the way".
            
            # Let's represent the ring as a linear path from 1 to N.
            # The path from l to t is blocked if r is between them.
            
            # Option 1: Move from l to t without crossing r.
            # There are two paths on a ring. One path is [l, l+1, ..., t] (mod N)
            # and the other is [l, l-1, ..., t] (mod N).
            
            # To find the shortest path not passing through r:
            # We can simulate the movement or use the property that the only way to 
            # move on the ring is to go around the "gap" created by r.
            # The available parts are {1, ..., N} \ {r}. This is a linear chain of N-1 parts.
            # We need the distance between l and t in this linear chain.
            
            # Map the ring to a linear sequence starting from (r % N) + 1
            # Example: N=6, r=4. Sequence: 5, 6, 1, 2, 3.
            # The distance is the absolute difference of their indices in this sequence.
            
            # Correct way to calculate distance on a ring with a hole at r:
            # The distance is the shortest path in the graph where edges (i, i+1) 
            # exist if neither endpoint is r (Wait, that's wrong. The hole is at r, 
            # so we can't land on r).
            # The edges are (i, i+1) where the destination is not r.
            
            # Let's find the distance from l to t in the graph where node r is removed.
            # The graph is a cycle minus one node, which is a path.
            # In a path, the distance between two nodes is unique.
            
            # To find the distance from l to t in a ring of N nodes with r removed:
            # We can use the coordinates relative to r.
            # Let's transform coordinates such that r becomes 0 (or N).
            # New coordinate x' = (x - r - 1) % N + 1
            # Now r is at position N (or 0). The remaining nodes are 1 to N-1.
            # The distance is simply |x'_l - x'_t|.
            
            l_prime = (l - r - 1) % n + 1
            t_prime = (t - r - 1) % n + 1
            dist = abs(l_prime - t_prime)
            total_ops += dist
            l = t
            
        else: # h == 'R'
            # Move right hand from r to t, left hand l stays fixed
            r_prime = (r - l - 1) % n + 1
            t_prime = (t - l - 1) % n + 1
            dist = abs(r_prime - t_prime)
            total_ops += dist
            r = t

    print(total_ops)

if __name__ == "__main__":
    solve()