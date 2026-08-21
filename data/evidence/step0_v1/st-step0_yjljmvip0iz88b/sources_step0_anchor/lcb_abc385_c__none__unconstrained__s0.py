import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # We need to find the maximum number of buildings that:
    # 1. Have the same height H
    # 2. Are spaced at equal intervals D
    
    # We can iterate over all unique heights present in the array.
    # For each height, we find all indices where that height occurs.
    # Then we check all possible intervals between those indices.
    
    # However, a simpler approach using comprehensions:
    # Try every possible starting index 'i'
    # Try every possible interval 'd' (from 1 to N)
    # For a fixed i and d, count how many buildings at i, i+d, i+2d... have the same height.
    
    # To avoid loops, we use nested comprehensions.
    # We evaluate the length of the sequence for every valid (i, d) pair.
    # For a fixed start i and interval d, the buildings are at indices:
    # i, i+d, i+2d, ... as long as index < N.
    # They must all have height H = h[i].
    
    # We use a helper logic: for a fixed i and d, 
    # the number of buildings is the length of the longest prefix of 
    # [h[i], h[i+d], h[i+2d], ...] that equals h[i].
    
    # Since we can't use while loops, we can pre-generate the sequences
    # and use a trick with itertools.takewhile or a list comprehension 
    # that checks the condition.
    
    # Actually, the simplest way is:
    # For every i (start) and d (interval):
    # Count k such that h[i + k*d] == h[i] for all 0 <= k < count.
    # But the condition is "the chosen buildings" must have the same height.
    # It doesn't say we can't skip buildings of different heights.
    # Wait, "arranged at equal intervals" means if we pick indices p1, p2, ..., pk,
    # then p2-p1 = p3-p2 = ... = pk-p_{k-1} = d.
    # And h[p1] = h[p2] = ... = h[pk].
    
    # So for a fixed start i and interval d:
    # We check indices i, i+d, i+2d... 
    # We count how many of these have height == h[i].
    # IMPORTANT: The problem says "the chosen buildings are arranged at equal intervals".
    # This means if we choose indices {p, p+d, p+2d, ..., p+(k-1)d}, 
    # they must all have the same height.
    # It does NOT say we can't have a building of the same height at p + 0.5d.
    # It just means the set of indices we pick must form an arithmetic progression.
    
    # For a fixed i and d, we want to find the largest k such that
    # h[i], h[i+d], ..., h[i+(k-1)d] all have the same height.
    # This is NOT correct. We can pick ANY subset that forms an AP.
    # If we pick indices i, i+d, i+2d, we just need h[i] == h[i+d] == h[i+2d].
    # We don't care if h[i+0.5d] is the same or different.
    
    # Correct logic:
    # For every starting index i in 0...N-1:
    #   For every interval d in 1...N-1:
    #     Count how many j in {i, i+d, i+2d, ...} have h[j] == h[i].
    #     BUT, the condition is that the CHOSEN buildings are at equal intervals.
    #     If we choose indices {i, i+d, i+2d}, they are at equal intervals.
    #     If we choose {i, i+2d, i+4d}, they are also at equal intervals.
    #     So for a fixed i and d, we just need to count how many 
    #     indices in the sequence i, i+d, i+2d... have the height h[i].
    #     Wait, if we skip one, the interval changes.
    #     Example: Indices 0, 2, 4 are equal intervals (d=2).
    #     If h[0]=5, h[2]=5, h[4]=5, then we have 3 buildings.
    #     If h[0]=5, h[1]=5, h[2]=5, h[3]=5, h[4]=5, we could pick d=1 and get 5 buildings.
    
    # So the strategy:
    # For every pair of indices (i, j) where i < j and h[i] == h[j]:
    #   The interval is d = j - i.
    #   We check how many further indices i + 2d, i + 3d... also have height h[i].
    #   This is still a loop. 
    
    # Let's use the property that N is small (3000).
    # For a fixed height H, let indices be idx_list.
    # We want to find the longest AP in idx_list.
    # This is a classic problem, but we can't use loops.
    
    # Let's use a different approach:
    # For every possible interval d from 1 to N:
    #   For every starting position i from 0 to d-1:
    #     We have a sequence h[i], h[i+d], h[i+2d]...
    #     In this sequence, we want to find the most frequent height.
    #     Wait, that's not right. The chosen buildings must be at equal intervals.
    #     If we pick height H and interval d, we are looking at indices i, i+d, i+2d...
    #     We can only pick indices that actually have height H.
    #     If the sequence is [H, X, H, H], and we pick interval d,
    #     we can't just "skip" X and keep the interval d.
    #     The indices would be p, p+d, p+2d.
    #     So we are looking for the longest contiguous block of the same height 
    #     in the sequence h[i], h[i+d], h[i+2d]...
    #     NO, that's also wrong. The problem says "the chosen buildings are arranged at equal intervals".
    #     It doesn't say we can't have other buildings of the same height in between.
    #     It means we pick a set of indices {p, p+d, p+2d, ..., p+(k-1)d} 
    #     such that h[p] = h[p+d] = ... = h[p+(k-1)d].
    
    # Correct interpretation:
    # Maximize k such that there exist p, d where h[p] = h[p+d] = ... = h[p+(k-1)d].
    
    # Implementation without loops:
    # We can iterate over all possible d (1 to N) and all possible p (0 to N-1).
    # For a fixed p and d, we want to find the largest k such that 
    # h[p] == h[p+d] == ... == h[p+(k-1)d].
    # This is equivalent to finding the length of the prefix of the sequence
    # h[p], h[p+d], h[p+2d]... that consists of the same value h[p].
    
    # But we can't use while loops to find the prefix.
    # However, we can use a list comprehension to get the sequence 
    # and then find where it first differs.
    # Or even simpler:
    # For a fixed p and d, the number of buildings is:
    # k = (number of indices j = p + m*d < N such that h[j] == h[p])
    # WAIT: The condition is "the chosen buildings are arranged at equal intervals".
    # This means the indices are p, p+d, p+2d... 
    # It does NOT require that the buildings in between (like p + 0.5d) 
    # are NOT the same height. It just requires that the ones we PICK are.
    # And it requires that the ones we pick are spaced by d.
    # So if we pick indices {p, p+d, p+2d}, they are at equal intervals.
    # We just need h[p] == h[p+d] == h[p+2d].
    # To maximize the number of buildings, for a fixed p and d,
    # we just count how many m >= 0 satisfy p + m*d < N and h[p + m*d] == h[p].
    # BUT, the indices must be p, p+d, p+2d... 
    # If h[p] == H, h[p+d] == H, h[p+2d] == X, h[p+3d] == H,
    # we cannot pick {p, p+d, p+3d} because the intervals are d and 2d.
    # We must pick a subset of the sequence p, p+d, p+2d... 
    # that are themselves at equal intervals.
    # That means we pick a new interval d' = m*d.
    # This is getting complex. Let's simplify.
    
    # The condition "chosen buildings are arranged at equal intervals" 
    # means the indices are p, p+d, p+2d, ..., p+(k-1)d for some d > 0.
    # All these must have the same height.
    # So for a fixed p and d, we want to find the largest k such that
    # h[p] = h[p+d] = h[p+2d] = ... = h[p+(k-1)d].
    # This is exactly the length of the contiguous block of identical heights 
    # starting at p in the sequence h[p], h[p+d], h[p+2d]...
    
    # To implement this without loops:
    # For each p in 0...N-1 and d in 1...N:
    #   Sequence S = [h[p + m*d] for m in range((N-1-p)//d + 1)]
    #   We want the length of the prefix of S that equals h[p].
    #   Since we can't use while, we can use a trick:
    #   The length of the prefix is the index of the first element != h[p].
    #   We can find this using a list comprehension and `next()`.
    
    # Let's refine:
    # result = max(
    #   (
    #     next((m for m, val in enumerate([h[p + m*d] for m in range((n-1-p)//d + 1)]) if val != h[p]), (n-1-p)//d + 1)
    #     for p in range(n)
    #     for d in range(1, n)
    #   ),
    #   default=1
    # )
    
    # However, the constraints to avoid for/while loops apply to the whole program.
    # I will use nested comprehensions and `max`.
    
    print(max(
        [
            next(
                (m for m, val in enumerate(
                    [h[p + m*d] for m in range((n - 1 - p) // d + 1)]
                ) if val != h[p]),
                (n - 1 - p) // d + 1
            )
            for p in range(n)
            for d in range(1, n)
        ] + [1]
    ))

if __name__ == "__main__":
    solve()