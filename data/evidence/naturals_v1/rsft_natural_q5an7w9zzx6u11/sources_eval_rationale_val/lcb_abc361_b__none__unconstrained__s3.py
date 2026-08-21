import sys

def solve():
    # Read all input values into a single list
    data = list(map(int, sys.stdin.read().split()))
    
    # Cuboid 1: (a, b, c) to (d, e, f)
    # Cuboid 2: (g, h, i) to (j, k, l)
    # Indices: 0 1 2 3 4 5  6 7 8 9 10 11
    
    # Two cuboids have a positive intersection volume if and only if
    # their projections on all three axes (x, y, z) have a positive intersection length.
    # For an axis, intervals [x1, x2] and [x3, x4] overlap with positive length if:
    # max(x1, x3) < min(x2, x4)
    
    # X-axis: a=0, d=3, g=6, j=9
    # Y-axis: b=1, e=4, h=7, k=10
    # Z-axis: c=2, f=5, i=8, l=11
    
    has_positive_volume = (
        max(data[0], data[6]) < min(data[3], data[9]) and
        max(data[1], data[7]) < min(data[4], data[10]) and
        max(data[2], data[8]) < min(data[5], data[11])
    )
    
    print("Yes" if has_positive_volume else "No")

if __name__ == "__main__":
    solve()