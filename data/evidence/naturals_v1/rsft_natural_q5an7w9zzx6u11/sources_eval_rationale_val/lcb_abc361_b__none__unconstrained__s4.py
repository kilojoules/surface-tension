import sys

def solve():
    # Read all input values from stdin
    data = list(map(int, sys.stdin.read().split()))
    
    # Cuboid 1: (a, b, c) to (d, e, f)
    # Cuboid 2: (g, h, i) to (j, k, l)
    # We group the boundaries by axis: 
    # X: (a, d) and (g, j)
    # Y: (b, e) and (h, k)
    # Z: (c, f) and (i, l)
    
    # Extract boundaries
    # C1: data[0:3] (min), data[3:6] (max)
    # C2: data[6:9] (min), data[9:12] (max)
    
    # Two intervals [x1, x2] and [x3, x4] have a positive intersection length 
    # if and only if max(x1, x3) < min(x2, x4).
    
    # We create pairs of (min1, max1, min2, max2) for each axis
    axes = [
        (data[0], data[3], data[6], data[9]),   # X axis
        (data[1], data[4], data[7], data[10]),  # Y axis
        (data[2], data[5], data[8], data[11])   # Z axis
    ]
    
    # Check if the intersection is positive for all axes
    # Using a list comprehension to evaluate the condition for each axis
    # and all() to ensure every axis overlaps.
    result = all([max(min1, min2) < min(max1, max2) for min1, max1, min2, max2 in axes])
    
    if result:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()