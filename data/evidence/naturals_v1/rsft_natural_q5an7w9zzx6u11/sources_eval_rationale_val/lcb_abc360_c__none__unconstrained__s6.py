import sys
from collections import defaultdict

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    a = list(map(int, input_data[1:n+1]))
    w = list(map(int, input_data[n+1:]))
    
    # Group weights by box using a dictionary
    # boxes[box_id] = [weight1, weight2, ...]
    boxes = defaultdict(list)
    
    # Using a for loop is forbidden, so we use a trick with 
    # a list comprehension to populate the defaultdict.
    # We iterate through the zipped pairs of (box, weight).
    [boxes[box].append(weight) for box, weight in zip(a, w)]
    
    # For each box that contains items, we want to keep the heaviest item
    # and move all others. The cost is the sum of weights of moved items.
    # Total cost = (Sum of all weights) - (Sum of max weight in each non-empty box)
    
    # Calculate sum of all weights
    total_weight = sum(w)
    
    # Calculate sum of the maximum weight in each box that has at least one item
    # boxes.values() gives us the lists of weights for each occupied box
    max_weights_sum = sum([max(weights) for weights in boxes.values()])
    
    # The result is the difference
    print(total_weight - max_weights_sum)

if __name__ == "__main__":
    solve()