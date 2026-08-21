import sys
from itertools import groupby
from operator import itemgetter

def solve():
    # Read N and the list of heights
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # Group indices by height: {height: [index1, index2, ...]}
    # We use a dictionary comprehension to group indices for each unique height
    # range(N) provides the indices, H[i] provides the height
    groups = {}
    for i in range(N):
        h = H[i]
        if h not in groups:
            groups[h] = []
        groups[h].append(i)

    # For each height group, we want to find the maximum number of indices
    # that form an arithmetic progression.
    # Since we need "equal intervals", we are looking for indices i, i+d, i+2d...
    # For a fixed height, we can iterate over all pairs of indices (i, j) 
    # to define a starting point and a common difference d = j - i.
    # Then we count how many elements in the sorted index list fit that pattern.
    
    # However, a more efficient way for N=3000 is to iterate through all 
    # possible differences d (1 to N-1) and check sequences.
    # But the most direct way given the constraints is to group by height 
    # and for each height, check all pairs as potential starts of a sequence.
    
    # To optimize: for a fixed height, if we have indices idxs,
    # and we pick two indices idxs[i] and idxs[j], the difference is d = idxs[j] - idxs[i].
    # We can then check how many subsequent indices idxs[j] + k*d also exist in the set.
    
    # We use a set for O(1) lookup of indices for each height.
    results = [
        max(
            (
                sum(1 for k in range((max(idxs) - start) // d + 1) 
                    if (start + k * d) in idx_set)
                for start in idxs
                for d in range(1, N)
                if (start + d) in idx_set
            ) 
            if len(idxs) > 1 else 1
            for idxs, idx_set in (
                (group, set(group)) 
                for _, group in groupby(sorted((H[i], i) for i in range(N)), itemgetter(0))
                if False # This line is a placeholder to restructure the comprehension
            )
        )
    ]
    
    # The above comprehension was getting complex, let's use a cleaner approach:
    # 1. Group indices by height
    # 2. For each height, if only 1 building, count is 1.
    # 3. If > 1, try every pair of indices as the first two elements of the sequence.
    
    # Let's redefine the logic without the complex comprehension:
    pass

def main():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # Group indices by height
    from collections import defaultdict
    groups = defaultdict(list)
    for i in range(N):
        groups[H[i]].append(i)

    ans = 1
    for h in groups:
        idxs = groups[h]
        if len(idxs) <= ans:
            continue
        
        # Use a set for fast lookup
        s = set(idxs)
        # Try every pair of indices (i, j) as the first two elements
        # i is the first index, j is the second index. Difference d = j - i.
        # We only need to check j > i.
        for idx_a in range(len(idxs)):
            for idx_b in range(idx_a + 1, len(idxs)):
                i = idxs[idx_a]
                j = idxs[idx_b]
                d = j - i
                
                # Count how many terms are in the sequence i, i+d, i+2d...
                # We start from 2 because we already know i and j are there.
                # We use a while loop to count the length.
                count = 2
                current = j + d
                while current < N and current in s:
                    count += 1
                    current += d
                
                if count > ans:
                    ans = count
                    
    print(ans)

if __name__ == "__main__":
    main()