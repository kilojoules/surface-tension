import sys

def solve():
    # Read all input values into a single list
    # Using map() to avoid explicit for-loops for parsing
    data = list(map(int, sys.stdin.read().split()))
    
    # Cuboid 1: (a, b, c) to (d, e, f)
    # Cuboid 2: (g, h, i) to (j, k, l)
    # Indices: 0 1 2 3 4 5  6 7 8 9 10 11
    
    # Two cuboids have a positive intersection volume if and only if
    # they overlap in all three dimensions (x, y, and z).
    # An overlap in one dimension (x1, x2) and (x3, x4) exists if:
    # max(x1, x3) < min(x2, x4)
    
    # We check the condition for X, Y, and Z axes.
    # X: data[0] to data[3] and data[6] to data[9]
    # Y: data[1] to data[4] and data[7] to data[10]
    # Z: data[2] to data[5] and data[8] to data[11]
    
    has_volume = (
        max(data[0], data[6]) < min(data[3], data[9]) and
        max(data[1], data[7]) < min(data[4], data[10]) and
        max(data[2], data[8]) < min(data[5], data[11])
    )
    
    # Using a conditional expression to determine output string
    print("Yes" if has_volume else "No")

if __name__ == "__main__":
    solve()