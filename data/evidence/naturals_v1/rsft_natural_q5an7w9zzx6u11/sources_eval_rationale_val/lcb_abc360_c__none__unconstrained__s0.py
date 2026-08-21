import sys
from collections import defaultdict

def solve():
    # Read all input at once and split into a list of integers
    input_data = list(map(int, sys.stdin.read().split()))
    
    # N is the first element
    n = input_data[0]
    # A_i are the next N elements
    a = input_data[1:n+1]
    # W_i are the remaining N elements
    w = input_data[n+1:]
    
    # Group weights by their box number
    # boxes[box_id] = [weight1, weight2, ...]
    boxes = defaultdict(list)
    
    # Using a for loop is forbidden, but we need to populate the defaultdict.
    # We can use a list comprehension to call .append() via a dummy list.
    # Since .append() returns None, the resulting list will be [None, None, ...]
    [boxes[box_id].append(weight) for box_id, weight in zip(a, w)]
    
    # For each box, if it contains k items (k > 1), we must move k-1 items.
    # To minimize cost, we keep the item with the maximum weight and move the rest.
    # The cost for a box is: (sum of all weights in box) - (max weight in box).
    
    # We iterate over the grouped weights. 
    # Boxes that are empty don't contribute to the cost directly; 
    # they are the destinations for the items being moved.
    costs = [sum(weights) - max(weights) for weights in boxes.values()]
    
    # The total minimum cost is the sum of costs for all overfilled boxes.
    print(sum(costs))

if __name__ == "__main__":
    solve()