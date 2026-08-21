import sys

# Increase recursion depth for deep trees/sequences
sys.setrecursionlimit(300000)

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j"
    # means Building j must be a prefix maximum of the sequence H[i+1...N].
    # For a fixed i, we want to count j > i such that H[j] > max(H[i+1...j-1]).
    
    # To solve this for all i efficiently, we can use a Segment Tree or 
    # a Divide and Conquer approach. 
    # However, a simpler observation: we are counting elements to the right 
    # that are larger than all elements between them and index i.
    # This is equivalent to counting elements in the "upper hull" of the 
    # sequence starting from i+1.
    
    # Since we cannot use loops, we use recursion and mapping.
    # We use a Divide and Conquer approach: count_visible(l, r)
    # For the range [l, r], we find the maximum element at index 'm'.
    # Elements to the right of 'm' that are visible from 'l' are those 
    # that are visible from 'm' (since H[m] is the boss).
    # Elements between l and m that are visible from l are handled by the 
    # recursive call.
    
    # To implement this without loops, we use a Segment Tree to find the 
    # index of the maximum element in a range.
    
    tree = [0] * (4 * n)
    
    def build(node, start, end):
        if start == end:
            tree[node] = start
            return
        mid = (start + end) // 2
        build(2 * node, start, mid)
        build(2 * node + 1, mid + 1, end)
        # Store index of the maximum height
        if h[tree[2 * node]] > h[tree[2 * node + 1]]:
            tree[node] = tree[2 * node]
        else:
            tree[node] = tree[2 * node + 1]

    def query(node, start, end, l, r):
        if r < start or end < l:
            return -1
        if l <= start and end <= r:
            return tree[node]
        mid = (start + end) // 2
        left_idx = query(2 * node, start, mid, l, r)
        right_idx = query(2 * node + 1, mid + 1, end, l, r)
        if left_idx == -1: return right_idx
        if right_idx == -1: return left_idx
        return left_idx if h[left_idx] > h[right_idx] else right_idx

    build(1, 0, n - 1)

    def count_visible(i, l, r):
        if l > r:
            return 0
        # Find index of max element in range [l, r]
        m = query(1, 0, n - 1, l, r)
        # The max element H[m] is always visible from i.
        # Any element j > m is visible from i if it is visible from m.
        # Any element j < m is visible from i if it is visible from i in [l, m-1].
        return 1 + count_visible(i, l, m - 1) + count_visible(m, m + 1, r)

    # Use map to avoid for-loop for the final output
    results = map(lambda i: count_visible(i, i + 1, n - 1), range(n))
    print(*(results))

if __name__ == "__main__":
    solve()