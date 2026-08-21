import sys
from itertools import permutations

def solve():
    # Read input and parse N, S, T
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    # Parse line segments into a list of tuples ((Ax, Ay), (Cx, Cy))
    segments = [
        ((int(input_data[3 + 2*i]), int(input_data[4 + 2*i])),
         (int(input_data[5 + 2*i]), int(input_data[6 + 2*i])))
        for i in range(N)
    ]

    def dist(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    # Pre-calculate the time to print each segment
    # print_times[i] is the time taken to move from one end to the other at speed T
    print_times = [dist(seg[0], seg[1]) / T for seg in segments]

    # We need to try all permutations of segments and all possible directions for each segment.
    # A direction is represented by whether we start at endpoint 0 or endpoint 1.
    # There are N! permutations and 2^N direction combinations.
    # For N=6, 6! * 2^6 = 720 * 64 = 46,080, which is well within limits.
    
    # Generate all permutations of segment indices
    perms = permutations(range(N))
    
    # For a given permutation and a choice of directions, calculate total time.
    # directions: a tuple of 0 or 1. If 0, segment i is printed from endpoint 0 to 1.
    # If 1, segment i is printed from endpoint 1 to 0.
    
    # We use a helper to calculate the cost for a specific permutation and direction set.
    def calc_total_time(p, dirs):
        # Start at (0, 0)
        curr_pos = (0, 0)
        total_time = 0.0
        
        for idx in p:
            # Determine start and end points based on direction
            p0, p1 = segments[idx]
            start_pt = p0 if dirs[p.index(idx)] == 0 else p1
            end_pt = p1 if dirs[p.index(idx)] == 0 else p0
            
            # Time to move to start point at speed S + time to print at speed T
            total_time += dist(curr_pos, start_pt) / S
            total_time += print_times[idx]
            curr_pos = end_pt
            
        return total_time

    # To avoid the O(N) index lookup inside the loop, we can pre-process the 
    # segments into a list based on the permutation.
    
    # We use a list comprehension to iterate through all permutations and all 2^N direction vectors.
    # Since we cannot use loops, we use a nested comprehension.
    # For each permutation, we try all 2^N direction combinations.
    
    # Optimization: Instead of 2^N, we can use a recursive-like structure or 
    # simply map a function. But since we must avoid loops, we use a 
    # comprehension that generates all binary strings of length N.
    
    all_dirs = [tuple((i >> j) & 1 for j in range(N)) for i in range(1, <<N)] # Wait, << is not allowed
    # Correct way to get 2^N without loops:
    # We can use a list comprehension to generate all binary tuples.
    
    # Let's redefine the logic to fit in a single expression.
    # We want: min(total_time(perm, dirs)) for perm in perms, dirs in directions
    
    # To generate dirs without a loop:
    # We can use a list comprehension: [( (i>>j)&1 for j in range(N) ) for i in range(2**N)]
    
    # Final expression:
    result = min(
        sum(
            (dist(
                (segments[p[i]][0] if (dirs[i] == 0) else segments[p[i]][1]), 
                (segments[p[i-1]][1] if (dirs[i-1] == 0) else segments[p[i-1]][0]) 
                if i > 0 else (0, 0)
            ) / S) 
            + print_times[p[i]]
            for i in range(N)
        )
        for p in permutations(range(N))
        for dirs in [tuple((i >> j) & 1 for j in range(N)) for i in range(1 << N)]
    )

    print(f"{result:.20f}")

# The above logic uses 'range' and 'sum' which are allowed. 
# However, the 'for i in range(N)' inside the sum is technically a generator expression.
# Let's refine it to ensure it strictly follows the "no for/while loops" rule 
# (comprehensions are explicitly allowed).

def solve_final():
    input_data = sys.stdin.read().split()
    if not input_data: return
    N, S, T = map(int, input_data[:3])
    segs = [((int(input_data[3+2*i]), int(input_data[4+2*i])), 
             (int(input_data[5+2*i]), int(input_data[6+2*i]))) for i in range(N)]
    
    dist = lambda p1, p2: ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
    p_times = [dist(s[0], s[1])/T for s in segs]
    
    # We use a helper to calculate the cost of a sequence of (start, end) points
    # sequence: [(s1, e1), (s2, e2), ...]
    # cost = dist((0,0), s1)/S + len1/T + dist(e1, s2)/S + len2/T ...
    
    # Generate all possible ordered pairs of endpoints for each segment
    # For segment i, options are (segs[i][0], segs[i][1]) or (segs[i][1], segs[i][0])
    options = [ [(segs[i][0], segs[i][1]), (segs[i][1], segs[i][0])] for i in range(N) ]
    
    # We need to try all permutations of segments, and for each segment, one of the 2 options.
    # Using a generator to avoid explicit loops.
    
    # To handle the "no loops" constraint strictly, we use recursion-like 
    # structures via list comprehensions or map/reduce.
    # But since N is small (6), we can just use a comprehension that 
    # iterates through all permutations and all combinations of options.
    
    # We can use a trick to generate all combinations of options without a loop:
    # For N=6, we can just nest 6 comprehensions, but that's not generic.
    # Instead, we can use a list comprehension that iterates through 2^N.
    
    ans = min(
        sum(
            (dist(
                (p_opts[i][0] if (d >> i) & 1 == 0 else p_opts[i][1]), 
                (p_opts[i-1][1] if (d >> (i-1)) & 1 == 0 else p_opts[i-1][0]) 
                if i > 0 else (0, 0)
            ) / S) 
            + p_times[p[i]]
            for i in range(N)
        )
        for p in permutations(range(N))
        for d in range(1 << N)
        for p_opts in [ [segs[idx] for idx in p] ]
    )
    
    # Wait, the logic above with p_opts and d is slightly flawed because 
    # p_opts is already permuted, so d should index into the permuted list.
    # Let's fix:
    
    ans = min(
        sum(
            (dist(
                (segs[p[i]][0] if (d >> i) & 1 == 0 else segs[p[i]][1]), 
                (segs[p[i-1]][1] if (d >> (i-1)) & 1 == 0 else segs[p[i-1]][0]) 
                if i > 0 else (0, 0)
            ) / S) 
            + p_times[p[i]]
            for i in range(N)
        )
        for p in permutations(range(N))
        for d in range(1 << N)
    )
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve_final()