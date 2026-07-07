import sys

data = sys.stdin.read().split()
sx, sy, tx, ty = int(data[0]), int(data[1]), int(data[2]), int(data[3])

# Normalize each point to the left cell of its horizontal domino (x+y even)
sx -= (sx + sy) & 1
tx -= (tx + ty) & 1

# Transform to hex-lattice coordinates
Ps, Qs = (sx + sy) // 2, (sx - sy) // 2
Pt, Qt = (tx + ty) // 2, (tx - ty) // 2
dP, dQ = Pt - Ps, Qt - Qs

# Hex distance with allowed unit moves (1,0),(0,1),(1,1) and negatives
same_sign = (dP >= 0) == (dQ >= 0)
ans = max(abs(dP), abs(dQ)) if same_sign else abs(dP) + abs(dQ)
print(ans)
