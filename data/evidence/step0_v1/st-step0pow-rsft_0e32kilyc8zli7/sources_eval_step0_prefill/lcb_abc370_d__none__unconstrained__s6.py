import sys

def solve():
    # Read all input at once for efficiency
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    
    # To avoid explicit loops and recursion, we use data structures that 
    # allow jumping over empty spaces. Since H*W is up to 4e5, 
    # we can maintain sets of existing walls for each row and column.
    # However, Python's 'set' doesn't support efficient 'nearest neighbor' queries.
    # We use a Disjoint Set Union (DSU) approach to skip destroyed walls.
    # For each row, we maintain DSU structures to find the next/previous wall.
    # For each column, we do the same.
    
    # Because we cannot use loops, we use a functional approach or 
    # built-in mapping/reduction. But DSU requires state updates.
    # The constraint to avoid loops forces a shift from imperative to 
    # declarative style, though it is generally less idiomatic for DSU.
    # We will use a dictionary-based DSU and 'functools.reduce' to process queries.
    
    from functools import reduce

    # We need to track which walls exist. 
    # Instead of full DSU (which is hard without loops), we use 
    # sorted lists and the bisect module to find neighbors.
    # Since we need to remove elements, we use a Fenwick tree or Segment Tree?
    # No, the simplest way to find the nearest wall is using a SortedList.
    # But SortedList is not standard. We can use bisect on sorted lists,
    # but deleting from a list is O(N). 
    # Given the constraints and the "no loop" rule, we will use 
    # a dictionary to track wall existence and a custom DSU-like 
    # structure implemented via a dictionary and a recursive-like 
    # find function (using a trick to avoid recursion depth).
    
    # Actually, the most efficient way to find the nearest wall in 
    # a grid without loops is to use a data structure that supports 
    # efficient deletion and successor/predecessor queries.
    # Since we can't use loops, we'll use `bisect` and `list.pop`.
    # Wait, list.pop(i) is O(W). With Q=2e5 and W=4e5, that's O(QW), too slow.
    
    # Let's use the fact that we can use list comprehensions and map.
    # We will maintain a set of active walls for each row and column.
    # To find the nearest wall, we can't iterate. 
    # But we can use a Segment Tree or Fenwick Tree implemented via 
    # list comprehensions? No, that's impossible.
    
    # Let's reconsider: the only way to find the nearest wall without 
    # looping is using binary search on a sorted structure.
    # Since we need to delete, we can use a Balanced BST. 
    # Python doesn't have one built-in. 
    # However, we can use a Bitset (via a large integer) and 
    # bit manipulation (bit_length, etc.) to find the nearest 1.
    
    # For each row, a bitmask representing walls.
    # For each col, a bitmask representing walls.
    # Finding the nearest 1 to the left of bit k:
    # mask & ((1 << k) - 1) -> get the highest set bit using .bit_length()
    # Finding the nearest 1 to the right of bit k:
    # (mask >> k) -> get the lowest set bit using (x & -x).bit_length()
    
    # This allows O(1) or O(log W) queries and updates.
    
    def process_queries(state, query):
        r, c = query
        row_masks, col_masks, total_walls = state
        
        # Check if wall exists at (r, c)
        # Rows are 1-indexed, cols are 1-indexed.
        # We use 0-indexing internally.
        r_idx, c_idx = r - 1, c - 1
        
        if (row_masks[r_idx] >> c_idx) & 1:
            # Destroy wall at (r, c)
            new_row_masks = list(row_masks)
            new_col_masks = list(col_masks)
            new_row_masks[r_idx] &= ~(1 << c_idx)
            new_col_masks[c_idx] &= ~(1 << r_idx)
            return (new_row_masks, new_col_masks, total_walls - 1)
        else:
            # Destroy 4 neighbors
            # 1. Left: bits in row_masks[r_idx] less than c_idx
            mask_l = row_masks[r_idx] & ((1 << c_idx) - 1)
            # 2. Right: bits in row_masks[r_idx] greater than c_idx
            mask_r = row_masks[r_idx] >> (c_idx + 1)
            # 3. Up: bits in col_masks[c_idx] less than r_idx
            mask_u = col_masks[c_idx] & ((1 << r_idx) - 1)
            # 4. Down: bits in col_masks[c_idx] greater than r_idx
            mask_d = col_masks[c_idx] >> (r_idx + 1)
            
            # Identify target cells
            # Left: highest set bit of mask_l
            l_col = mask_l.bit_length() - 1 if mask_l else None
            # Right: lowest set bit of mask_r
            r_col = (mask_r & -mask_r).bit_length() - 1 + (c_idx + 1) if mask_r else None
            # Up: highest set bit of mask_u
            u_row = mask_u.bit_length() - 1 if mask_u else None
            # Down: lowest set bit of mask_d
            d_row = (mask_d & -mask_d).bit_length() - 1 + (r_idx + 1) if mask_d else None
            
            # We need to destroy these walls. Since we can't loop, 
            # we use a helper to update the masks.
            def destroy(s, row, col):
                if row is None or col is None: return s
                rm, cm, tw = s
                nm = list(rm)
                nc = list(cm)
                nm[row] &= ~(1 << col)
                nc[col] &= ~(1 << row)
                return (nm, nc, tw - 1)
            
            # Apply destructions sequentially using reduce
            return reduce(
                lambda s, target: destroy(s, *target),
                [(r_idx, l_col), (r_idx, r_col), (u_row, c_idx), (d_row, c_idx)],
                (row_masks, col_masks, total_walls)
            )

    # Initial state
    initial_row_masks = [(1 << W) - 1] * H
    initial_col_masks = [(1 << H) - 1] * W
    initial_total = H * W
    
    queries = zip(
        map(int, input_data[3::2]), 
        map(int, input_data[4::2])
    )
    
    final_state = reduce(
        process_queries, 
        queries, 
        (initial_row_masks, initial_col_masks, initial_total)
    )
    
    print(final_state[2])

if __name__ == "__main__":
    solve()