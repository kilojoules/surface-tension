import sys
from itertools import permutations

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    segments = [
        (int(input_data[3 + 2*i]), int(input_data[4 + 2*i]), 
         int(input_data[5 + 2*i]), int(input_data[6 + 2*i]))
        for i in range(N)
    ]

    def dist(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    # Pre-calculate lengths of segments
    seg_lengths = [dist((s[0], s[1]), (s[2], s[3])) for s in segments]
    
    # We need to try all permutations of segments and all 2^N orientations
    # A state is defined by (current_permutation, orientations)
    # But since N is small (<= 6), we can use a generator expression inside min()
    
    # For a given permutation and orientation:
    # orientation i=0: start at (Ai, Bi), end at (Ci, Di)
    # orientation i=1: start at (Ci, Di), end at (Ai, Bi)
    
    # We use a helper to calculate total time for a specific sequence
    def calc_time(perm, orientations):
        # Initial position
        curr_pos = (0, 0)
        total_time = 0.0
        
        for idx, orient in zip(perm, orientations):
            s = segments[idx]
            p_start = (s[0], s[1]) if orient == 0 else (s[2], s[3])
            p_end = (s[2], s[3]) if orient == 0 else (s[0], s[1])
            
            # Time to move to start (at speed S) + time to print (at speed T)
            total_time += dist(curr_pos, p_start) / S
            total_time += seg_lengths[idx] / T
            curr_pos = p_end
            
        return total_time

    # Generate all permutations of indices
    all_perms = permutations(range(N))
    # Generate all possible orientation combinations (0 or 1 for each segment)
    # Since we can't use loops, we use a list comprehension to generate all 2^N tuples
    all_orients = [
        tuple((perm_orient >> i) & 1 for i in range(N))
        for perm_orient in range(1 << N)
    ]

    # The result is the minimum time across all permutations and all orientation settings
    # We use a generator expression to avoid explicit for-loops
    ans = min(
        calc_time(p, o) 
        for p in all_perms 
        for o in all_orients
    )
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()