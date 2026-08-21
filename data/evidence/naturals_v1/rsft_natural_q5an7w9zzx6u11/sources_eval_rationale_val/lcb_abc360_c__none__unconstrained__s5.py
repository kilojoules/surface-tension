import sys
from collections import defaultdict

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract N and the lists A and W
    # Since we cannot use loops, we use next() to get N and slicing for the rest
    # However, since we need to handle the stream, we convert to a list first
    data = list(input_data)
    n = data[0]
    a = data[1:n+1]
    w = data[n+1:]

    # Group weights by box: box_contents[box_id] = [weight1, weight2, ...]
    # We use a dictionary and a trick with a list comprehension to populate it
    # Since we can't loop, we use a helper function or a reduce-like approach.
    # Actually, the most idiomatic "no-loop" way to group is using a dictionary
    # and updating it via a comprehension that evaluates a side-effect.
    
    # To avoid explicit for-loops, we use a dictionary and the fact that 
    # we can iterate via map/comprehensions.
    box_contents = defaultdict(list)
    [box_contents[box_id].append(weight) for box_id, weight in zip(a, w)]

    # For each box that has more than one item, we keep the heaviest item 
    # and move the others. The cost is the sum of all weights in the box 
    # minus the maximum weight in that box.
    # We only care about boxes that actually contain items.
    
    # Calculate cost for each box: sum(weights) - max(weights)
    # This is done via a list comprehension over the dictionary values.
    costs = [sum(weights) - max(weights) for weights in box_contents.values()]
    
    # The final answer is the sum of these costs.
    print(sum(costs))

if __name__ == "__main__":
    solve()