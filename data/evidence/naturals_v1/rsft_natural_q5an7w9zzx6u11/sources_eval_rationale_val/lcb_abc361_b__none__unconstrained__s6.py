import sys

def solve():
    # Read all input values into a single list
    data = list(map(int, sys.stdin.read().split()))
    
    # Cuboid 1: (a, b, c) to (d, e, f)
    # Cuboid 2: (g, h, i) to (j, k, l)
    # Indices: 0 1 2 3 4 5  6 7 8 9 10 11
    
    # Two cuboids have a positive volume intersection if and only if
    # their projections onto all three axes have an intersection with positive length.
    # For an axis, intervals [min1, max1] and [min2, max2] overlap with positive length if:
    # max(min1, min2) < min(max1, max2)
    
    # X-axis: [data[0], data[3]] and [data[6], data[9]]
    # Y-axis: [data[1], data[4]] and [data[7], data[10]]
    # Z-axis: [data[2], data[5]] and [data[8], data[11]]
    
    overlap_x = max(data[0], data[6]) < min(data[3], data[9])
    overlap_y = max(data[1], data[7]) < min(data[4], data[10])
    overlap_z = max(data[2], data[8]) < min(data[5], data[11])
    
    if overlap_x and overlap_y and overlap_z:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()