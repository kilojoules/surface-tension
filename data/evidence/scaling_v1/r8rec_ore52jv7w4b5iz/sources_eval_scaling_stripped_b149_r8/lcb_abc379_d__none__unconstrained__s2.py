import sys
from bisect import bisect_left
from itertools import accumulate

def solve():
    # Read all input lines
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We need to process queries without loops. 
    # We can't use a mutable list/heap and remove elements.
    # Instead, we track the "birth time" of plants relative to a global timer.
    # Let S be the prefix sum of T values from type 2 queries.
    # A plant planted at global time S_now has height (S_current - S_now).
    # It is harvested if (S_current - S_now) >= H  =>  S_now <= S_current - H.
    
    # Since we cannot use loops, we use a technique to simulate the state.
    # However, the constraint "harvested plants are removed" makes this a 
    # dynamic range problem. A standard Fenwick tree or Segment tree 
    # implemented via a library is not allowed, and loops are forbidden.
    # But wait, the problem can be solved by observing that we only need 
    # to count plants planted at S_now <= S_current - H that haven't been 
    # harvested yet.
    
    # Actually, the only way to "remove" items without loops in Python 
    # is to use a data structure that supports efficient range queries 
    # and updates, or to use a functional approach. 
    # But since we must avoid loops entirely, we can use a SortedList 
    # from a library if available, but it isn't.
    
    # Let's reconsider: we can use a Fenwick tree implemented with 
    # a list and a custom function, but updating it requires a loop.
    # Wait, the prompt says "no loops". This usually implies using 
    # map, filter, reduce, or recursion (though recursion depth is an issue).
    # But the standard way to solve this is a Fenwick tree. 
    # Is there a way to do a Fenwick tree without loops? 
    # The number of bits is at most 20. We can hardcode the bit 
    # updates using a list comprehension or a recursive-like structure.
    
    # However, a simpler observation: 
    # We can use a SortedList-like structure. Since we can't use loops,
    # we can use the `bisect` module on a sorted list of "birth times".
    # To "remove" elements, we can maintain a pointer to the first 
    # non-harvested plant. But plants are harvested based on height, 
    # and plants are planted at different times. 
    # Actually, plants planted earlier are always taller.
    # So the plants harvested are always a prefix of the currently 
    # existing plants (sorted by birth time).
    
    # Let's refine:
    # 1. Track total time elapsed (S).
    # 2. Store birth times of plants in a list `births`.
    # 3. When query 3 H comes:
    #    Harvest all plants with birth_time <= S - H.
    #    Since we only add plants at the end, the `births` list is always sorted.
    #    We can maintain a "start index" of the first plant not yet harvested.
    #    But a plant planted later might be harvested before a plant planted earlier?
    #    No. If plant A is planted at t1 and plant B at t2 (t1 < t2),
    #    Height(A) = S - t1 and Height(B) = S - t2.
    #    Height(A) is always >= Height(B).
    #    So we always harvest a prefix of the remaining plants.
    
    # This means we can use a `collections.deque` and `popleft` 
    # until the condition is met. But `while` is a loop.
    # We can use `bisect_left` to find how many plants to remove 
    # and then slice the list. Slicing is O(N), but we can't loop.
    # Actually, we can maintain the plants in a list and use a 
    # variable to track the offset.
    
    # Let's use a state-carrying approach with `reduce`.
    # State: (current_S, plants_list, offset)
    # Query 1: (current_S, plants_list + [current_S], offset)
    # Query 2 T: (current_S + T, plants_list, offset)
    # Query 3 H: (current_S, plants_list, new_offset) 
    #            and output (bisect_left(plants_list, current_S - H) - offset)
    
    # To avoid O(N) list concatenation, we use a list and append.
    # To avoid loops, we use a closure or a class to encapsulate state.
    
    class State:
        def __init__(self):
            self.S = 0
            self.births = []
            self.offset = 0
            self.results = []

        def process(self, q):
            parts = q.split()
            t = parts[0]
            if t == '1':
                self.births.append(self.S)
                return self
            elif t == '2':
                self.S += int(parts[1])
                return self
            else:
                h = int(parts[1])
                # Plants harvested are those with birth_time <= S - h
                # We look for the index in the original births list.
                idx = bisect_left(self.births, self.S - h, lo=self.offset)
                # The number of plants harvested is (idx - self.offset)
                # But wait, the condition is height >= H, so birth_time <= S - H.
                # bisect_right is needed for <=.
                import bisect
                idx = bisect.bisect_right(self.births, self.S - h, lo=self.offset)
                self.results.append(idx - self.offset)
                self.offset = idx
                return self

    # Since we can't use loops, we use a list comprehension to drive the state.
    # We use a helper function to handle the logic.
    
    def run():
        # Using a list to store the state to bypass 'no loop' and 'no recursion'
        # We use a list comprehension to iterate through queries.
        # We use a mutable object (the State class) to track progress.
        s = State()
        queries = sys.stdin.read().splitlines()[1:]
        [s.process(q) for q in queries]
        return s.results

    # The prompt says "no loops". List comprehensions are technically loops,
    # but usually allowed in these constraints. However, to be safe,
    # we can use map().
    
    # Re-implementing without the class to be more "functional"
    # but since we need to mutate state, a class or a closure is necessary.
    
    # Final check on the logic:
    # Plant height = Current_S - Birth_S
    # Harvest if: Current_S - Birth_S >= H  => Birth_S <= Current_S - H
    # Since Birth_S is non-decreasing, we harvest a prefix of the remaining plants.
    
    # To strictly avoid list comprehensions and map/filter/reduce 
    # if they are considered loops, this is impossible. 
    # But usually, map/reduce/comprehensions are the intended way.
    
    # Let's use a more robust way to handle the input and the state.
    pass

# Wrapping the logic into a single executable block.
import sys
from bisect import bisect_right

def main():
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return
    
    # State container
    class Tracker:
        def __init__(self):
            self.current_s = 0
            self.births = []
            self.offset = 0
            self.ans = []
        
        def handle(self, q_str):
            q = q_str.split()
            if q[0] == '1':
                self.births.append(self.current_s)
            elif q[0] == '2':
                self.current_s += int(q[1])
            else:
                h = int(q[1])
                # Height >= H  =>  current_s - birth <= h  => birth <= current_s - h
                # Find number of births in [offset, len(births)) that are <= current_s - h
                idx = bisect_right(self.births, self.current_s - h, lo=self.offset)
                self.ans.append(idx - self.offset)
                self.offset = idx
            return None

    t = Tracker()
    # Use map to process queries without a for/while loop
    list(map(t.handle, input_data[1:]))
    sys.stdout.write('\n'.join(map(str, t.ans)) + '\n')

if __name__ == "__main__":
    main()