import sys

def solve():
    # Read all input values into a single list
    data = list(map(int, sys.stdin.read().split()))
    
    # Cuboid 1: (a, b, c) to (d, e, f)
    # Cuboid 2: (g, h, i) to (j, k, l)
    a, b, c, d, e, f = data[0:6]
    g, h, i, j, k, l = data[6:12]
    
    # Two cuboids have a positive intersection volume if and only if
    # their projections onto all three axes (x, y, z) overlap by a positive length.
    # For two intervals [x1, x2] and [x3, x4], the overlap is positive if:
    # max(x1, x3) < min(x2, x4)
    
    has_positive_volume = (
        max(a, g) < min(d, j) and
        max(b, h) < min(e, k) and
        max(c, i) < min(f, l)
    )
    
    print("Yes" if has_positive_volume else "No")

if __name__ == "__main__":
    solve()