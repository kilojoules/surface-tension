import sys

def solve():
    # Read all input values into a single list
    data = list(map(int, sys.stdin.read().split()))
    
    # Cuboid 1: (a, b, c) to (d, e, f)
    # Cuboid 2: (g, h, i) to (j, k, l)
    # Indices: 0 1 2 3 4 5  6 7 8 9 10 11
    
    # Two cuboids have a positive intersection volume if and only if 
    # their projections onto all three axes (x, y, z) overlap by a positive length.
    # For an axis, intervals [min1, max1] and [min2, max2] overlap if:
    # max(min1, min2) < min(max1, max2)
    
    # X-axis: a=0, d=3, g=6, j=9
    # Y-axis: b=1, e=4, h=7, k=10
    # Z-axis: c=2, f=5, i=8, l=11
    
    overlap_x = max(data[0], data[6]) < min(data[3], data[9])
    overlap_y = max(data[1], data[7]) < min(data[4], data[10])
    overlap_z = max(data[2], data[8]) < min(data[5], data[11])
    
    if overlap_x and overlap_y and overlap_z:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()