import sys

def solve():
    # Read all input and split into integers
    data = list(map(int, sys.stdin.read().split()))
    if not data:
        return
    
    N = data[0]
    H = data[1:]

    # The condition "no building taller than Building j between i and j"
    # means that for a fixed i, we are looking for the number of j > i
    # such that H[k] < H[j] for all k where i < k < j.
    #
    # This is equivalent to saying that Building j is a "visible" building
    # when looking to the right from Building i, but specifically 
    # defined by the height of the target building j, not the source i.
    #
    # Let's rephrase: j satisfies the condition if H[j] > max(H[i+1]...H[j-1]).
    # This means H[j] must be a prefix maximum of the sequence H[i+1...N].
    #
    # To solve this for all i efficiently:
    # We can use a Monotonic Stack to find for each j, how many i < j it "covers".
    # Specifically, Building j is the prefix maximum for all i such that
    # there is no k in (i, j) with H[k] > H[j].
    # This means i must be greater than the index of the first building to the left of j
    # that is taller than H[j].
    # Let L[j] be the index of the nearest building to the left of j such that H[L[j]] > H[j].
    # If no such building exists, L[j] = 0 (using 1-based indexing).
    # Then for a fixed j, it satisfies the condition for all i such that L[j] <= i < j.
    # The number of such i is j - L[j].
    #
    # We want to find for each i, the count of j > i.
    # This is equivalent to: for each j from 1 to N, add 1 to the range [L[j], j-1].
    # We can use a difference array to handle these range updates.

    # L array to store the index of the nearest taller building to the left
    # Using 0-based indexing internally: L[j] is the index of the first k < j with H[k] > H[j]
    L = [-1] * N
    stack = []

    for j in range(N):
        while stack and H[stack[-1]] < H[j]:
            stack.pop()
        if stack:
            L[j] = stack[-1]
        stack.append(j)

    # Difference array for range updates
    # We want to add 1 to range [L[j] + 1, j - 1] (0-indexed)
    # Note: if L[j] = -1, the range is [0, j-1].
    # The condition is: i < j and max(H[i+1...j-1]) < H[j].
    # This is true for i in {L[j], L[j]+1, ..., j-1}.
    # Wait, let's re-verify:
    # If L[j] is the index of the first building to the left taller than H[j],
    # then for any i from L[j] to j-1, the buildings between i and j are 
    # H[i+1...j-1], all of which are smaller than H[j].
    # Example: H = [2, 1, 4, 3, 5], j=2 (H[j]=4). L[2]=-1.
    # i can be 0, 1. (Buildings 1 and 2).
    # For i=0: H[1...1] = [1]. 1 < 4. OK.
    # For i=1: H[2...1] = []. OK.
    # So j=2 contributes to c_0 and c_1.
    # The range of i is [L[j], j-1] if we use 0-based indexing for i.
    # But we must ensure i < j.
    
    # Let's use a difference array `diff` of size N+1
    diff = [0] * (N + 1)
    for j in range(N):
        # Range is [L[j] + 1, j] ? No.
        # If L[j] = -1, i can be 0, 1, ..., j-1.
        # If L[j] = 0, i can be 1, 2, ..., j-1.
        # The number of i's is j - (L[j] + 1) + 1 = j - L[j].
        # The indices are i \in {L[j]+1, ..., j-1} PLUS the case where i is the 
        # building immediately to the left of the "taller" one?
        # Let's use the logic: j satisfies the condition for i if i < j AND 
        # there is no k such that i < k < j and H[k] > H[j].
        # This means H[k] < H[j] for all k from i+1 to j-1.
        # This is true if i >= L[j].
        # Since we also need i < j, i is in {L[j], L[j]+1, ..., j-1}.
        # However, L[j] is the index of the building taller than H[j].
        # If i = L[j], the buildings between i and j are H[L[j]+1 ... j-1].
        # By definition of L[j], all these are < H[j]. So i = L[j] is allowed.
        # But we must ensure i >= 0.
        # So i ranges from max(0, L[j]) to j-1.
        
        # Correction: If L[j] is the index of the nearest building to the left 
        # that is TALLER than H[j], then for any i such that L[j] <= i < j,
        # the buildings strictly between i and j (indices k: i < k < j)
        # are all smaller than H[j].
        # Example: H = [10, 2, 3, 4], j=3 (H[j]=4). L[3]=0 (H[0]=10).
        # i can be 0, 1, 2.
        # i=0: k \in {1, 2}, H[1]=2, H[2]=3. Both < 4. OK.
        # i=1: k \in {2}, H[2]=3. < 4. OK.
        # i=2: k \in {}, OK.
        # So i is in [L[j], j-1] if L[j] is the index of the first building > H[j].
        # If no such building exists, L[j] = -1, but i starts at 0, so [0, j-1].
        # Wait, if L[j] = 0, then H[0] > H[j]. The buildings between i=0 and j are H[1...j-1].
        # These are all < H[j]. So i=0 is actually allowed.
        # The only building that can block j is one that is TALLER than H[j].
        # Let L[j] be the index of the nearest building to the left such that H[L[j]] > H[j].
        # Then for any i < L[j], the building at L[j] is between i and j and is taller than H[j].
        # So we must have i >= L[j].
        # Thus i \in {L[j], L[j]+1, ..., j-1}.
        # But we must also have i >= 0.
        # So i \in {max(0, L[j]), ..., j-1}.
        
        # Let's re-verify with Sample 1: 2 1 4 3 5
        # j=0: H=2, L=-1. i \in {0...-1} -> none.
        # j=1: H=1, L=0. i \in {0...0} -> i=0.
        # j=2: H=4, L=-1. i \in {0...1} -> i=0, 1.
        # j=3: H=3, L=2. i \in {2...2} -> i=2.
        # j=4: H=5, L=-1. i \in {0...3} -> i=0, 1, 2, 3.
        # Counts: i=0: {1, 2, 4}, i=1: {2, 4}, i=2: {3, 4}, i=3: {4}, i=4: {}
        # Results: 3, 2, 2, 1, 0. Matches Sample 1.
        
        left = L[j]
        right = j - 1
        if left <= right:
            # We need to increment range [max(0, left), right]
            # Since we can't use loops, use difference array.
            # Note: if left is -1, max(0, -1) is 0.
            start = 0 if left < 0 else left
            diff[start] += 1
            diff[right + 1] -= 1

    # Compute prefix sums to get the final counts
    result = [0] * N
    current = 0
    for i in range(N):
        current += diff[i]
        result[i] = current

    print(*(result))

if __name__ == "__main__":
    solve()