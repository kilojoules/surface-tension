import sys
from collections import defaultdict
from itertools import chain

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # N is the first element
    n = int(input_data[0])
    # A_i are the next N elements
    a = map(int, input_data[1:n+1])
    # W_i are the remaining N elements
    w = map(int, input_data[n+1:])
    
    # Group weights by their box number
    # Using a dictionary of lists to store weights of items in each box
    boxes = defaultdict(list)
    
    # Since we cannot use for-loops, we use a trick with a list comprehension
    # to populate the defaultdict. The side-effect of .append() is used.
    # We zip A and W to associate each item's box with its weight.
    [boxes[box].append(weight) for box, weight in zip(a, w)]
    
    # For each box, if it contains k items (k > 1), we must move k-1 items.
    # To minimize cost, we keep the heaviest item and move the k-1 lightest.
    # 1. Sort the weights in each box.
    # 2. Take all but the last (heaviest) item.
    # 3. Sum all these 'moved' weights.
    
    # We iterate over the values of the dictionary.
    # sorted(weights) sorts in ascending order.
    # sorted(weights)[:-1] slices all but the maximum weight.
    
    redundant_weights = [
        sorted(weights)[:-1] 
        for weights in boxes.values() 
        if len(weights) > 1
    ]
    
    # Use itertools.chain to flatten the list of lists and sum the result.
    total_cost = sum(chain.from_iterable(redundant_weights))
    
    print(total_cost)

if __name__ == "__main__":
    solve()