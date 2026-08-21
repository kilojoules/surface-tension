```python
import sys
from bisect import bisect_left
from itertools import accumulate
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We need to process queries in a way that we can efficiently 
    # count and remove plants.
    # Let S be the cumulative sum of T values from type 2 queries.
    # A plant planted at time 't' (where S_t is the cumulative growth at that time)
    # has height (S_now - S_t) at current time.
    # Condition: S_now - S_t >= H  =>  S_t <= S_now - H.
    
    # First, parse the queries into a list of tuples
    # Since queries have different lengths, we use a custom parser.
    queries = []
    idx = 1
    while idx < len(input_data):
        q_type = input_data[idx]
        if q_type == '1':
            queries.append((1, 0))
            idx += 1
        elif q_type == '2':
            queries.append((2, int(input_data[idx+1])))
            idx += 2
        else:
            queries.append((3, int(input_data[idx+1])))
            idx += 2

    # Calculate cumulative growth over time
    # growth_at_step[i] is the total T added before query i
    growth_deltas = [q[1] if q[0] == 2 else 0 for q in queries]
    growth_timeline = list(accumulate(growth_deltas, initial=0))
    # growth_timeline[i] is the sum of T for queries 0...i-1
    # The actual growth at the moment of query i is growth_timeline[i]
    # Wait, the growth happens DURING query 2. 
    # Let's refine: 
    # Query i is type 2 T: growth increases by T.
    # Query i is type 3 H: height is (CurrentTotalGrowth - GrowthAtPlanting).
    
    # Correct logic for growth_timeline:
    # We want the total growth prefix sum BEFORE query i.
    # But type 2 increases growth for all EXISTING plants.
    # Let's use a different approach:
    # Maintain a sorted list of 'birth_growth' values.
    # A plant born at query i has birth_growth = current_total_growth.
    # It is harvested if current_total_growth - birth_growth >= H.
    
    # To handle the "removal" and "counting" efficiently:
    # We can't use a simple list because removals are slow.
    # However, we only remove plants that satisfy a threshold.
    # Since birth_growth is non-decreasing (we only add plants as time goes on),
    # the plants are naturally sorted by birth_growth.
    # We can use a deque or a sorted list and binary search for the threshold.
    # Since we only remove from the left (smallest birth_growth), 
    # a deque or simply tracking the index of the first active plant works.
    
    # Wait, the plants are NOT necessarily sorted by birth_growth if we 
    # just add them. But we ARE adding them chronologically.
    # So the list of birth_growths of existing plants is always sorted.
    
    # Let's simulate:
    # current_growth: total T encountered so far
    # plants: a sorted list of growth values at the time of planting
    # For type 3 H: harvest plants where growth_timeline[now] - birth_growth >= H
    # birth_growth <= growth_timeline[now] - H
    
    # Since we need to remove elements from the middle/start, 
    # and Q is 2*10^5, we can use a Fenwick tree or Segment tree 
    # over the indices of plants created, but that's complex.
    # Actually, we only remove plants from the "left" of the sorted birth_growth list.
    # Because birth_growth is non-decreasing, the condition birth_growth <= threshold
    # will always apply to a prefix of the current plants.
    
    # We can use a sorted list (via a specialized data structure) or 
    # since we only remove prefixes, a simple list with a pointer (index).
    # But we aren't removing a prefix of ALL plants ever created, 
    # only a prefix of EXISTING plants. 
    # Actually, if we keep a list of birth_growths of all plants ever created,
    # and a way to mark them as "harvested", we can use a Fenwick tree 
    # to count how many are still active in the prefix.
    
    # Let's refine:
    # 1. Identify all indices i where query i is type 1.
    # 2. For each such i, calculate G_i = total growth before query i.
    # 3. For query type 3 H at index j:
    #    Threshold = (total growth at index j) - H.
    #    We need count of i < j such that G_i <= Threshold AND plant i is not yet harvested.
    #    Since G_i is non-decreasing with i, the condition G_i <= Threshold 
    #    is satisfied by all i in range [0, max_idx] where max_idx is found by bisect.
    #    The number of harvested plants is (number of active plants in [0, max_idx]).
    
    # To track active plants:
    # Use a Fenwick tree over the indices of type-1 queries.
    # When plant i is created, update(i, 1).
    # When plants in range [0, max_idx] are harvested, 
    # we need to sum the Fenwick range and then set those indices to 0.
    # To avoid O(N) updates, we can keep track of the `last_harvested_idx`.
    # Since we only harvest prefixes of the G_i array, we only need to 
    # update the Fenwick tree for indices from `last_harvested_idx + 1` to `max_idx`.
    
    # Wait, if we only harvest prefixes, we don't even need a Fenwick tree.
    # We just need to know how many plants were created up to `max_idx` 
    # and subtract how many were already harvested.
    
    # Let's trace:
    # Plants created at indices: p1, p2, p3... (where p is the query index)
    # Their birth_growths: G_{p1}, G_{p2}, G_{p3}... (this is sorted!)
    # Query 3 H at index j:
    # Threshold = CurrentGrowth_j - H
    # Find largest k such that G_{pk} <= Threshold.
    # Plants harvested: all active plants in {p1, ..., pk}.
    # Since we always harvest a prefix of the plants, the number of harvested
    # plants is simply (k - last_k), where last_k is the number of plants
    # harvested in previous type-3 queries.
    
    # This is much simpler! No Fenwick tree needed.
    
    # Implementation:
    # 1. Calculate prefix sums of T.
    # 2. Store G_i for every type-1 query.
    # 3. For type-3, binary search for threshold in G list, update last_k.
    
    # Let's double check: does "harvest all plants with height >= H" 
    # always remove a prefix of the current plants?
    # Height = CurrentGrowth - BirthGrowth.
    # Height >= H  =>  BirthGrowth <= CurrentGrowth - H.
    # Since BirthGrowth is non-decreasing, yes, it's always a prefix of the 
    # plants sorted by birth date.
    
    # Final Algorithm:
    # - growth_at_query = accumulate([T if type==2 else 0 for q in queries])
    # - birth_growths = [growth_at_query[i] for i, q in enumerate(queries) if q[0] == 1]
    # - current_growth_at_query = growth_at_query (shifted by 1)
    # - For query type 3 H at index i:
    #   threshold = growth_at_query[i+1] - H
    #   k = bisect_right(birth_growths, threshold)
    #   # We must only count plants created BEFORE query i.
    #   # So k = min(k, number of plants created before query i)
    #   # Answer = max(0, k - last_k)
    #   # last_k = max(last_k, k)
    
    # Wait, the growth_at_query logic:
    # Query 0: Type 1 -> growth 0. birth_growths = [0]
    # Query 1: Type 2 T=15 -> growth becomes 15.
    # Query 2: Type 1 -> growth 15. birth_growths = [0, 15]
    # Query 3: Type 3 H=10 -> threshold = 15 - 10 = 5.
    # bisect_right([0, 15], 5) returns 1.
    # Answer: 1 - 0 = 1. last_k = 1.
    # Query 4: Type 2 T=20 -> growth becomes 35.
    # Query 5: Type 3 H=20 -> threshold = 35 - 20 = 15.
    # bisect_right([0, 15], 15) returns 2.
    # Answer: 2 - 1 = 1. last_k = 2.
    
    # One detail: the number of plants created before query i.
    # Let's use a list comprehension to get the index of each type-1 query.
    # plant_indices = [i for i, q in enumerate(queries) if q[0] == 1]
    # For query i, the number of plants created so far is 
    # bisect_right(plant_indices, i).
    
    # Let's refine the growth timeline:
    # growth_timeline = accumulate([T if type==2 else 0 for q in queries])
    # This gives the growth AFTER each query.
    # For query i:
    # If type 1: birth_growth = growth_timeline[i-1] (or 0 if i=0)
    # If type 3: current_growth = growth_timeline[i]
    
    # Correct growth timeline:
    # T_vals = [q[1] if q[0] == 2 else 0 for q in queries]
    # pref = accumulate(T_vals)
    # growth_after = list(pref)
    # growth_before = [0] + growth_after[:-1]
    
    # birth_growths = [growth_before[i] for i, q in enumerate(queries) if q[0] == 1]
    # plant_creation_query_idx = [i for i, q in enumerate(queries) if q[0] == 1]
    
    # For query i of type 3 H:
    # current_g = growth_after[i]
    # threshold = current_g - H
    # # Plants created before or at query i:
    # plants_so_far_count = bisect_right(plant_creation_query_idx, i)
    # # Plants satisfying height condition:
    # k = bisect_right(birth_growths, threshold, hi=plants_so_far_count)
    # ans = max(0, k - last_k)
    # last_k = max(last_k, k)
    
    # Let's implement this.
    
    T_vals = [q[1] if q[0] == 2 else 0 for q in queries]
    growth_after = list(accumulate(T_vals))
    growth_before = [0] + growth_after[:-1]
    
    birth_growths = [growth_before[i] for i, q in enumerate(queries) if q[0] == 1]
    plant_creation_query_idx = [i for i, q in enumerate(queries) if q[0] == 1]
    
    # To avoid loops, we use a generator/map to process type-3 queries
    # But we need to maintain state (last_k). 
    # We can use a custom class or a function with a mutable state.
    
    def process_queries(state, q_idx):
        q_type, q_val = queries[q_idx]
        if q_type == 3:
            current_g = growth_after[q_idx]
            threshold = current_g - q_val
            plants_so_far = bisect_right(plant_creation_query_idx, q_