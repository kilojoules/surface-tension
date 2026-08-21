import sys
from collections import defaultdict

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    a = list(map(int, input_data[1:n+1]))
    w = list(map(int, input_data[n+1:]))
    
    # Group weights by box: box_contents[box_id] = [weight1, weight2, ...]
    # Using a dictionary and a list comprehension to group weights
    # Since we cannot use for-loops, we use a trick with a dictionary 
    # and the list.append method inside a list comprehension.
    box_contents = defaultdict(list)
    [box_contents[box_id].append(weight) for box_id, weight in zip(a, w)]
    
    # For boxes containing more than one item, we keep the heaviest item 
    # (to avoid moving it) and move the others.
    # The cost for a box is the sum of all weights minus the maximum weight.
    
    # We only care about boxes that actually have items.
    # Boxes with 0 items will be filled by items moved from boxes with >1 item.
    
    # Calculate cost: for each box, if it has k items, we move k-1 items.
    # The cheapest way is to keep the heaviest item in the box.
    costs = [
        sum(weights) - max(weights) 
        for weights in box_contents.values() 
        if len(weights) > 1
    ]
    
    # Output the total sum of costs
    print(sum(costs))

if __name__ == "__main__":
    solve()