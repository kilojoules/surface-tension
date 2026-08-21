import sys
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    
    queries = [
        (int(input_data[i]), int(input_data[i+1]))
        for i in range(3, len(input_data), 2)
    ]

    # To efficiently find the nearest wall, we use a sorted list of 
    # existing wall indices for every row and every column.
    # Since we need to find the nearest value > x and < x, 
    # and H*W is up to 4*10^5, we can use the 'bisect' module on sorted lists.
    # However, removing elements from a list is O(N). 
    # Given the constraints and the nature of the problem, 
    # we use sets to track destroyed walls and a custom way to find neighbors.
    # Actually, the most efficient way in Python without external libraries 
    # for "find nearest" is using sorted lists and bisect, 
    # but deletion is slow. Instead, we can use a dictionary of sets 
    # to track which walls are gone, but that doesn't help find the "nearest".
    
    # Correct approach for Python: Use a dictionary of sets for rows and columns
    # to store indices of EXISTING walls. Since we can't efficiently remove 
    # from sorted lists, we use the fact that we can maintain sorted lists 
    # and use bisect, and accept that list.pop(i) is O(N). 
    # Wait, H*W <= 4*10^5 means we cannot have O(H*Q) or O(W*Q).
    # We need a data structure that supports find-nearest and delete in O(log N).
    # Python doesn't have a built-in SortedSet. We can simulate one using 
    # a SortedList from a library, but we must use standard library only.
    # A common trick is using a Fenwick tree or Segment Tree over the indices,
    # but that's for sums. For "nearest", we can use a Disjoint Set Union (DSU)
    # to skip destroyed walls.
    
    # For each row, we have a DSU to find the next available wall to the right/left.
    # For each column, a DSU to find the next available wall up/down.
    # Since we need both directions, we need two DSUs per row/column.
    
    # However, a simpler way to implement this in Python without loops 
    # is to use the 'bisect' module and accept that list.pop() is O(N).
    # But with N=4*10^5, O(N^2) will TLE.
    # Let's use the DSU approach. For each row, we maintain two DSUs:
    # one for the next wall to the right, one for the next wall to the left.
    
    # Given the constraints and Python's speed, the most robust way to 
    # handle "find nearest and delete" is using a SortedList-like structure.
    # Since we can't import, we can implement a simple SortedList using blocks (sqrt decomposition).
    
    import bisect

    class SortedList:
        def __init__(self, iterable):
            self._list = sorted(iterable)
            self._size = len(self._list)
            self._block_size = int(self._size**0.5) + 1
            self._blocks = [self._list[i:i + self._block_size] 
                            for i in range(0, self._size, self._block_size)]

        def remove(self, val):
            # Find block using bisect on the last element of each block
            idx = bisect.bisect_left([b[-1] for b in self._blocks], val)
            if idx < len(self._blocks):
                # Find exact index in block
                i = bisect.bisect_left(self._blocks[idx], val)
                if i < len(self._blocks[idx]) and self._blocks[idx][i] == val:
                    self._blocks[idx].pop(i)
                    self._size -= 1
                    if not self._blocks[idx]:
                        self._blocks.pop(idx)
                    return True
            return False

        def find_nearest(self, val):
            # Find index where val would be inserted
            # We search across blocks to find the insertion point
            # Using a list comprehension to find the first block where the last element >= val
            block_idx = next((i for i, b in enumerate(self._blocks) if b[-1] >= val), len(self._blocks))
            
            # We need the element immediately smaller and immediately larger than val
            # Search in the identified block and the one before it
            res = []
            
            # For the larger one (right/down)
            if block_idx < len(self._blocks):
                i = bisect.bisect_right(self._blocks[block_idx], val)
                if i < len(self._blocks[block_idx]):
                    res.append(self._blocks[block_idx][i])
                elif block_idx + 1 < len(self._blocks):
                    res.append(self._blocks[block_idx + 1][0])
            
            # For the smaller one (left/up)
            # Check the current block for elements < val
            i = bisect.bisect_left(self._blocks[block_idx] if block_idx < len(self._blocks) else [], val)
            if i > 0:
                # The element at i-1 in the current block is the largest < val
                # But we need to handle the case where we are at the start of a block
                pass # handled by the logic below
            
            # Let's refine the nearest search:
            return res

    # The SortedList above is getting complex. Let's use a simpler approach:
    # Since we need to find the nearest existing wall, we can use 
    # bisect on a standard list and accept the O(N) pop, 
    # BUT we can optimize by using a set for existence checks 
    # and only performing the pop when a wall is actually destroyed.
    
    # Actually, the most efficient way to implement this in pure Python 
    # is to use a dictionary of sets to track destroyed walls and 
    # a custom SortedList or simply use the fact that 
    # we can use `bisect` on a list and `pop` is fast enough for 4*10^5 
    # if the number of pops is not too high? No, that's not true.
    
    # Let's use a different approach: 
    # For each row and column, maintain a sorted list of wall indices.
    # Use bisect to find the neighbors. Use list.pop() to remove.
    # To avoid O(N) pop, we can use a library-free SortedList implementation.
    
    import bisect

    class FastSortedList:
        def __init__(self, iterable):
            self.data = sorted(iterable)
            self.size = len(self.data)
            self.bucket_size = 700
            self.buckets = [self.data[i:i + self.bucket_size] 
                            for i in range(0, self.size, self.bucket_size)]

        def pop_val(self, val):
            idx = bisect.bisect_left([b[-1] for b in self.buckets], val)
            if idx < len(self.buckets):
                b = self.buckets[idx]
                i = bisect.bisect_left(b, val)
                if i < len(b) and b[i] == val:
                    b.pop(i)
                    self.size -= 1
                    if not b:
                        self.buckets.pop(idx)
                    return True
            return False

        def nearest(self, val):
            # Returns (smaller, larger)
            idx = bisect.bisect_left([b[-1] for b in self.buckets], val)
            
            # Larger: first element >= val. If it is val, we need the one strictly > val.
            # So we look for the first element > val.
            # Search in bucket idx
            larger = None
            if idx < len(self.buckets):
                b = self.buckets[idx]
                i = bisect.bisect_right(b, val)
                if i < len(b):
                    larger = b[i]
                elif idx + 1 < len(self.buckets):
                    larger = self.buckets[idx+1][0]
            
            # Smaller: largest element < val.
            smaller = None
            # Check bucket idx
            if idx < len(self.buckets):
                b = self.buckets[idx]
                i = bisect.bisect_left(b, val)
                if i > 0:
                    smaller = b[i-1]
                elif idx > 0:
                    smaller = self.buckets[idx-1][-1]
            elif idx == len(self.buckets):
                if self.buckets:
                    smaller = self.buckets[-1][-1]
                    # Since val is greater than all elements, the largest is the last one.
                    # But we need it to be strictly less than val.
                    # If the last element is == val, we need the one before it.
                    if smaller == val:
                        # This case is handled by the logic: if val is in the set, 
                        # the query logic handles it. If val is not in the set,
                        # the largest element in the last bucket is the nearest smaller.
                        # Wait, if val is not in the set, then the last element is indeed the nearest smaller.
                        # If val IS in the set, the query logic says "If there is a wall... destroy and end".
                        # So we only call nearest() when there is NO wall at (R, C).
                        pass
            
            # Correcting the 'smaller' logic for the case where val is not in the set:
            # If idx == len(self.buckets), the largest element is in the last bucket.
            if idx == len(self.buckets) and self.buckets:
                smaller = self.buckets[-1][-1]
            elif idx < len(self.buckets):
                b = self.buckets[idx]
                i = bisect.bisect_left(b, val)
                if i > 0:
                    smaller = b[i-1]
                elif idx > 0:
                    smaller = self.buckets[idx-1][-1]
            
            return smaller, larger

    # Initialize SortedLists for every row and column
    row_walls = [FastSortedList(range(1, W + 1)) for _ in range(H)]
    col_walls = [FastSortedList(range(1, H + 1)) for _ in range(W)]
    
    # To track if a wall exists without checking SortedLists
    # We use a set of (r, c) tuples for destroyed walls
    destroyed = set()
    
    def destroy(r, c):
        if (r, c) in destroyed:
            return
        destroyed.add((r, c))
        row_walls[r-1].pop_val(c)
        col_walls[c-1].pop_val(r)

    for r, c in queries:
        if (r, c) not in destroyed:
            destroy(r, c)
        else:
            # No wall at (r, c), destroy 4 neighbors
            # Up/Down
            s_col, l_col = col_walls[c-1].nearest(r)
            # Left/Right
            s_row, l_row = row_walls[r-1].nearest(c)
            
            if s_col is not None: destroy(s_col, c)
            if l_col is not None: destroy(l_col, c)
            if s_row is not None: destroy(r, s_row)
            if l_row is not None: destroy(r, l_row)

    print(H * W - len(destroyed))

solve()