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
    # instructions = [(H_i, T_i), ...]
    instructions = [
        (input_data[2 + 2*i], int(input_data[3 + 2*i]))
        for i in range(Q)
    ]

    # State: (left_hand_pos, right_hand_pos)
    # We want to find the minimum operations to reach the final state.
    # Since N and Q are small (100), we can use DP.
    # dp[i][l][r] = min operations after instruction i with hands at (l, r)
    # However, the problem guarantees that positions are uniquely determined.
    # Let's maintain the current positions (l, r) and calculate the distance.
    
    curr_l, curr_r = 1, 2
    total_ops = 0

    def get_dist(start, end, obstacle, n):
        # Distance from start to end on a ring of size n avoiding obstacle
        # There are two paths: clockwise and counter-clockwise.
        # A path is valid if the obstacle is not on it.
        
        # Path 1: Increasing index (with wrap around)
        # Path 2: Decreasing index (with wrap around)
        
        # To simplify, we can use a BFS to find the shortest path avoiding the obstacle
        # since N is very small (100).
        queue = [(start, 0)]
        visited = {start}
        idx = 0
        while idx < len(queue):
            node, dist = queue[idx]
            idx += 1
            if node == end:
                return dist
            
            # Neighbors on the ring
            for neighbor in [(node % n) + 1, ((node - 2) % n) + 1]:
                if neighbor != obstacle and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return float('inf')

    for h, t in instructions:
        if h == 'L':
            # Move left hand to t, right hand stays at curr_r
            dist = get_dist(curr_l, t, curr_r, N)
            total_ops += dist
            curr_l = t
        else:
            # Move right hand to t, left hand stays at curr_l
            dist = get_dist(curr_r, t, curr_l, N)
            total_ops += dist
            curr_r = t

    print(total_ops)

if __name__ == "__main__":
    solve()