import sys
from bisect import bisect_left

def solve():
    # Use fast I/O
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    ptr = 1
    
    # current_height_offset keeps track of the total height added by Type 2 queries.
    # When a plant is added (Type 1), its "birth height" relative to the offset 
    # is -current_height_offset.
    current_height_offset = 0
    
    # We store the birth heights of all existing plants in a sorted list.
    # Since Type 1 queries only add plants and Type 3 removes them, 
    # and we need to count plants with height >= H,
    # Height of plant = birth_height + current_height_offset.
    # Condition: birth_height + current_height_offset >= H  => birth_height >= H - current_height_offset.
    
    # Because we only add plants and remove the tallest ones (those with the smallest birth_heights),
    # a simple sorted list or a deque could work, but we need to remove from the "left" 
    # (the oldest plants).
    
    plants = []
    # To handle removals efficiently, we can use a pointer or a deque.
    # However, since we only remove plants that meet the height criteria and 
    # the height criteria is monotonic relative to the time they were planted,
    # the plants that are harvested will always be a prefix of the plants list 
    # if we store them in order of creation.
    
    # Let's refine:
    # Plants are created at times t1, t2, ...
    # Plant i height at time now = (Total T added from t_i to now).
    # Let S[t] be the prefix sum of T values.
    # Height of plant created at query q = S[current_q] - S[q].
    # Harvest if: S[current_q] - S[q] >= H  => S[q] <= S[current_q] - H.
    
    # We can store the S[q] values of existing plants in a list.
    # Since q increases, S[q] is non-decreasing.
    # The plants that satisfy the condition are those created earliest.
    
    birth_s_values = []
    head = 0 # pointer to the first plant not yet harvested
    
    results = []
    
    for _ in range(Q):
        q_type = int(input_data[ptr])
        ptr += 1
        
        if q_type == 1:
            # Plant height 0 means its S[q] is the current total offset
            birth_s_values.append(current_height_offset)
        elif q_type == 2:
            T = int(input_data[ptr])
            ptr += 1
            current_height_offset += T
        elif q_type == 3:
            H = int(input_data[ptr])
            ptr += 1
            
            # Condition: current_height_offset - S[q] >= H
            # S[q] <= current_height_offset - H
            threshold = current_height_offset - H
            
            # Find how many plants have S[q] <= threshold.
            # Since birth_s_values is naturally sorted (current_height_offset is non-decreasing),
            # we can use binary search to find the range of plants to harvest.
            
            # We only care about plants from index 'head' onwards.
            # Find the index of the first plant with S[q] > threshold.
            idx = bisect_left(birth_s_values, threshold + 1, lo=head)
            
            # Number of plants harvested is idx - head.
            results.append(str(idx - head))
            head = idx
            
    sys.stdout.write('\n'.join(results) + '\n')

if __name__ == "__main__":
    solve()