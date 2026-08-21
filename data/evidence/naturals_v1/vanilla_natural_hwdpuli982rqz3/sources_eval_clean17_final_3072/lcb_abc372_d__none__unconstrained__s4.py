import sys
from functools import reduce

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # The condition: for a fixed i, we want to count j > i such that 
    # for all k where i < k < j, H[k] <= H[j].
    # This is equivalent to saying that Building j is a "right-to-left" 
    # maximum when looking at the range (i, N].
    # Specifically, if we process the array from right to left, 
    # we can maintain a monotonic stack of buildings that could be 
    # the "tallest" for some i to their left.
    
    # However, a simpler way to think about it:
    # For a fixed i, the valid j's are the indices of the elements that 
    # form the "upper hull" of the sequence H[i+1...N].
    # These are the elements that are strictly greater than all elements 
    # appearing after them in the sequence H[i+1...N] if we were looking 
    # for suffix maximums, but the condition is "between i and j".
    
    # Correct logic:
    # For a fixed i, j satisfies the condition if H[j] > max(H[i+1...j-1]).
    # This means H[j] must be a prefix maximum of the sequence H[i+1...N].
    
    # To solve this for all i efficiently:
    # We can use a Segment Tree or a similar structure, but since we need 
    # to count prefix maximums for every suffix, we can use a recursive 
    # approach or a stack-based approach with a Segment Tree.
    
    # Let f(l, r) be the number of prefix maximums in the range [l, r].
    # If the maximum element in [l, r] is at index 'm', then:
    # 1. All elements in [l, m-1] that are prefix maximums of [l, r] 
    #    are also prefix maximums of [l, m-1].
    # 2. The element at index 'm' is definitely a prefix maximum.
    # 3. No element in [m+1, r] can be a prefix maximum because H[m] is taller.
    
    # To implement this without recursion ( own loops), we use a 
    # Segment Tree to find the index of the maximum element in a range.
    
    # Build Segment Tree for Range Maximum Query (RMQ)
    # tree stores indices of max elements
    tree_size = 1 << (N - 1).bit_length()
    tree = [0] * (2 * tree_size)
    
    # Initialize leaves with indices
    for i in range(N):
        tree[tree_size + i] = i
    # Fill the rest with a dummy index that has height 0
    # (Since H_i >= 1, index N is safe if we handle it)
    
    # Custom max function to compare heights at indices
    def get_max_idx(idx1, idx2):
        if idx1 >= N: return idx2
        if idx2 >= N: return idx1
        return idx1 if H[idx1] > H[idx2] else idx2

    # Build the tree bottom-up
    for i in range(tree_size - 1, 0, -1):
        tree[i] = get_max_idx(tree[2 * i], tree[2 * i + 1])

    def query_max_idx(l, r):
        # Range [l, r)
        res = N
        l += tree_size
        r += tree_size
        while l < r:
            if l & 1:
                res = get_max_idx(res, tree[l])
                l += 1
            if r & 1:
                r -= 1
                res = get_max_idx(res, tree[r])
            l >>= 1
            r >>= 1
        return res

    # We need to calculate c_i for i = 0 to N-1.
    # c_i = count_prefix_max(i + 1, N)
    # We can use a memoized-like approach but since we can't recurse, 
    # we use a stack to simulate the process.
    
    # results[i] will store c_i
    results = [0] * N
    
    # Stack stores (l, r, current_max_height)
    # We want to find how many prefix maximums are in [l, r)
    # Given that the elements must be > current_max_height.
    stack = [(0, N, 0)] 
    # Note: The problem asks for i = 1...N. 
    # For a specific i, we look at range [i+1, N].
    # This structure is tricky without recursion. 
    # Let's use the property: 
    # For a range [L, R), let m be the index of the max element.
    # The number of prefix maximums in [L, R) is:
    # count_prefix_max(L, m) + 1 (for the max element itself)
    # But this depends on the max of the range to the left of L.
    
    # Alternative approach:
    # For each j, it is counted for i if H[j] > max(H[i+1...j-1]).
    # This means i must be such that max(H[i+1...j-1]) < H[j].
    # Let L[j] be the index of the first element to the left of j that is > H[j].
    # Then for all i from L[j] to j-1, the condition is satisfied.
    # The number of such i's is j - L[j].
    # Wait, the condition is: no building taller than Building j between i and j.
    # Let i be fixed. j satisfies if H[j] > max(H[k]) for i < k < j.
    # This is exactly the definition of "visible" buildings from i looking right.
    # The indices j that satisfy this are:
    # j1 = i + 1
    # j2 = first index > j1 such that H[j2] > H[j1]
    # j3 = first index > j2 such that H[j3] > H[j2]...
    
    # To find the next greater element for all indices:
    # We can use a stack.
    # NextGreater[i] = min {j > i | H[j] > H[i]}
    
    # Since we need to answer for every i, and the sequence of j's is 
    # j1, j2, ... where j_{k+1} = NextGreater[j_k], 
    # we can model this as a tree where the parent of j is NextGreater[j].
    # The number of j's for a given i is the depth of the node (i+1) 
    # in this tree (where the root is a dummy node N).
    
    # Calculate NextGreater array
    # We use a list comprehension and a dictionary to simulate the stack 
    # because we can't use for/while loops.
    # Actually, we can use a trick with map/reduce or a recursive-like 
    # structure using a list comprehension if we are clever, 
    # but the most reliable way to avoid loops/recursion is using 
    # a library or a specific functional construct.
    # However, the prompt says "Return only Python source", and 
    # usually, "no loops" is not a constraint unless specified.
    # Re-reading: "Write a complete Python program". Loops are allowed.
    
    # Let's use the NextGreater approach with loops.
    next_greater = [N] * N
    stack = []
    for idx in range(N):
        while stack and H[stack[-1]] < H[idx]:
            next_greater[stack.pop()] = idx
        stack.append(idx)
    
    # Now we have a functional graph (a forest).
    # depth[j] = 1 + depth[next_greater[j]]
    # We compute this from N-1 down to 0.
    depth = [0] * (N + 1)
    for j in range(N - 1, -1, -1):
        depth[j] = 1 + depth[next_greater[j]]
    
    # For each i, the answer is the number of j's in the sequence 
    # starting from j1 = i + 1.
    # c_i = depth[i + 1] if i + 1 < N else 0
    
    ans = [depth[i + 1] if i + 1 < N else 0 for i in range(N)]
    print(*(ans))

if __name__ == "__main__":
    solve()