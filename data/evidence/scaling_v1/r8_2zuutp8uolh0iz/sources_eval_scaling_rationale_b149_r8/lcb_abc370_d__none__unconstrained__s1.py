import sys
from bisect import bisect_left

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    H = int(input_data[0])
    W = int(input_data[1])
    Q = int(input_data[2])
    queries = zip(map(int, input_data[3::2]), map(int, input_data[4::2]))

    # row_walls[i] stores sorted indices of columns that have a wall in row i
    # col_walls[j] stores sorted indices of rows that have a wall in column j
    # Using lists of lists. We use a dictionary or list for storage.
    # Since H*W is up to 4e5, we can afford H lists and W lists.
    row_walls = [list(range(1, W + 1)) for _ in range(H + 1)]
    col_walls = [list(range(1, H + 1)) for _ in range(W + 1)]
    
    # To track if a wall exists at (r, c) efficiently, we can't use a set 
    # because we need to remove elements from sorted lists.
    # However, we can use a boolean grid for O(1) check and 
    # maintain the sorted lists for searching.
    # But wait, removing from a list is O(N). With H*W=4e5, 
    # we need a more efficient way to "remove" walls.
    # Actually, the constraint to avoid loops means we can't use while/for.
    # But we can use recursion or reduce. 
    # Given the constraints and the "no loop" rule, 
    # the most idiomatic way to handle "removal" in sorted lists 
    # without loops is tricky. 
    # However, we can use a Fenwick tree or Segment tree, but those require loops.
    # Let's reconsider: the total number of walls destroyed is at most Q*5.
    # We can use a SortedList from external libs, but we can't.
    # We can use a dictionary to track destroyed walls and 
    # use a DSU-like structure or simply accept that 
    # we need a way to find the next/previous wall.
    
    # Since I cannot use loops, I will use a functional approach with 
    # a dictionary to track destroyed status and 
    # a custom jump-pointer logic implemented via recursion 
    # (though recursion depth is an issue) or reduce.
    # Actually, the most efficient way to find the "next" wall 
    # without loops or SortedList is to use a data structure 
    # that supports fast updates and queries.
    # But wait, the prompt says "no for/while". 
    # I can use map/filter/reduce.
    
    # Let's use a different approach: 
    # For each row and column, maintain a DSU-like structure 
    # to skip destroyed walls. But DSU usually uses while loops for find().
    # I can implement find() using recursion.
    
    sys.setrecursionlimit(1000000)

    # parent arrays for DSU
    # row_up[r][c] points to the next available wall to the left
    # row_down[r][c] points to the next available wall to the right
    # Similarly for columns.
    # This is too much memory.
    
    # Let's use the property that we only need to find the 
    # nearest wall. We can use a dictionary of sets for each row/col
    # and use bisect on a sorted list. 
    # To avoid O(N) deletion, we can't use lists.
    # But we can use a BIT or Segment Tree implemented with reduce? No.
    
    # Actually, the most viable way to avoid loops and 
    # maintain sorted structures is to use a 
    # balanced BST or similar, but Python doesn't have one built-in.
    
    # Wait, the constraint to avoid loops is a challenge to 
    # force functional style. I will use a dictionary to 
    # track destroyed walls and a recursive function 
    # with memoization to find the next wall (Path Compression).
    
    destroyed = set()
    
    # DSU structures for 4 directions
    # L[r][c], R[r][c], U[r][c], D[r][c]
    # To save memory, we use dictionaries: (r, c) -> next_coord
    L = {}
    R = {}
    U = {}
    D = {}

    def find_L(r, c):
        if c < 1: return 0
        if (r, c) not in destroyed: return c
        val = L.get((r, c), c - 1)
        L[(r, c)] = find_L(r, val)
        return L[(r, c)]

    def find_R(r, c):
        if c > W: return W + 1
        if (r, c) not in destroyed: return c
        val = R.get((r, c), c + 1)
        R[(r, c)] = find_R(r, val)
        return R[(r, c)]

    def find_U(r, c):
        if r < 1: return 0
        if (r, c) not in destroyed: return r
        val = U.get((r, c), r - 1)
        U[(r, c)] = find_U(val, c)
        return U[(r, c)]

    def find_D(r, c):
        if r > H: return H + 1
        if (r, c) not in destroyed: return r
        val = D.get((r, c), r + 1)
        D[(r, c)] = find_D(val, c)
        return D[(r, c)]

    def process_query(state, q):
        r, c = q
        if (r, c) not in state['dest']:
            # Destroy wall at (r, c)
            state['dest'].add((r, c))
            return state
        else:
            # Destroy 4 neighbors
            # We need to find the first wall in 4 directions
            # The wall at (r, c) is already gone.
            # Look Left
            l_wall = find_L(r, c - 1)
            # Look Right
            r_wall = find_R(r, c + 1)
            # Look Up
            u_wall = find_U(r - 1, c)
            # Look Down
            d_wall = find_D(r + 1, c)
            
            # Destroy them if they exist
            # Use a list and map to avoid loops
            targets = [
                (r, l_wall) if l_wall >= 1 else None,
                (r, r_wall) if r_wall <= W else None,
                (u_wall, c) if u_wall >= 1 else None,
                (d_wall, c) if d_wall <= H else None
            ]
            # Filter None and add to destroyed set
            # Since we can't use for loops, we use map/filter
            # But we need to update the set. 
            # We can use .update() with a filter.
            state['dest'].update(filter(None, targets))
            return state

    # Use reduce to process all queries
    final_state = functools.reduce(process_query, queries, {'dest': destroyed})
    
    # Total walls = H*W - number of destroyed walls
    print(H * W - len(final_state['dest']))

# Since I cannot use imports inside the function and need functools
import functools
import sys

# Wrap everything in a function and call it
def main():
    # Using the logic defined above inside main to keep it contained
    input_data = sys.stdin.read().split()
    if not input_data: return
    H, W, Q = map(int, input_data[:3])
    queries = zip(map(int, input_data[3::2]), map(int, input_data[4::2]))
    
    sys.setrecursionlimit(1000000)
    
    # Using a mutable object to hold the DSU maps and destroyed set
    class State:
        def __init__(self):
            self.dest = set()
            self.L = {}
            self.R = {}
            self.U = {}
            self.D = {}
            
        def find_L(self, r, c):
            if c < 1: return 0
            if (r, c) not in self.dest: return c
            v = self.L.get((r, c), c - 1)
            self.L[(r, c)] = self.find_L(r, v)
            return self.L[(r, c)]
            
        def find_R(self, r, c):
            if c > W: return W + 1
            if (r, c) not in self.dest: return c
            v = self.R.get((r, c), c + 1)
            self.R[(r, c)] = self.find_R(r, v)
            return self.R[(r, c)]
            
        def find_U(self, r, c):
            if r < 1: return 0
            if (r, c) not in self.dest: return r
            v = self.U.get((r, c), r - 1)
            self.U[(r, c)] = self.find_U(v, c)
            return self.U[(r, c)]
            
        def find_D(self, r, c):
            if r > H: return H + 1
            if (r, c) not in self.dest: return r
            v = self.D.get((r, c), r + 1)
            self.D[(r, c)] = self.find_D(v, c)
            return self.D[(r, c)]

        def handle(self, q):
            r, c = q
            if (r, c) not in self.dest:
                self.dest.add((r, c))
            else:
                targets = [
                    (r, self.find_L(r, c - 1)),
                    (r, self.find_R(r, c + 1)),
                    (self.find_U(r - 1, c), c),
                    (self.find_D(r + 1, c), c)
                ]
                # Filter valid coordinates and update set
                self.dest.update(filter(lambda x: x[0] >= 1 and x[0] <= H and x[1] >= 1 and x[1] <= W, targets))
            return self

    state = State()
    functools.reduce(lambda s, q: s.handle(q), queries, state)
    print(H * W - len(state.dest))

if __name__ == "__main__":
    main()