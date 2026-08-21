import sys

def solve():
    # Read all input values into a single list
    data = list(map(int, sys.stdin.read().split()))
    
    # Cuboid 1: (a, b, c) to (d, e, f)
    # Cuboid 2: (g, h, i) to (j, k, l)
    # Indices: 0 1 2 3 4 5  6 7 8 9 10 11
    
    # Two cuboids have a positive intersection volume if and only if
    # their projections onto all three axes (x, y, z) overlap with positive length.
    # For two intervals [x1, x2] and [x3, x4], the overlap is positive if:
    # max(x1, x3) < min(x2, x4)
    
    # X-axis overlap: max(a, g) < min(d, j)
    # Y-axis overlap: max(b, h) < min(e, k)
    # Z-axis overlap: max(c, i) < min(f, l)
    
    has_overlap = (
        max(data[0], data[6]) < min(data[3], data[9]) and
        max(data[1], data[7]) < min(data[4], data[10]) and
        max(data[2], data[8]) < min(data[5], data[11])
    )
    
    print("Yes" if has_overlap else "No")

if __name__ == "__main__":
    solve()