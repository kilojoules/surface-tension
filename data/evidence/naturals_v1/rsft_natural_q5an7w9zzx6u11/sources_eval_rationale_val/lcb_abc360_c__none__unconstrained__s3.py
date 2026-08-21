import sys
from collections import defaultdict

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    A = input_data[1:N+1]
    W = map(int, input_data[N+1:])
    
    # Group weights by box using a dictionary of lists
    # We use a comprehension to build the groups
    # Since we can't use for-loops, we use a trick with defaultdict and a map
    boxes = defaultdict(list)
    
    # To avoid for-loops, we use a helper function and map to populate the defaultdict
    # map(lambda x: boxes[x[0]].append(x[1]), zip(A, W)) 
    # However, map is lazy in Python 3, so we wrap it in a list or use a deque
    # But wait, the constraint says "no for/while loops". 
    # We can use a list comprehension to execute the append operations.
    [boxes[a].append(w) for a, w in zip(A, W)]
    
    # For each box, if it has k items, we keep the heaviest one and move the other k-1.
    # The cost is the sum of all weights in the box minus the maximum weight in that box.
    # We only care about boxes that have at least one item.
    
    # Calculate (sum of weights - max weight) for every box that contains items.
    # We use a list comprehension to iterate over the dictionary values.
    costs = [sum(weights) - max(weights) for weights in boxes.values()]
    
    # The total cost is the sum of these individual box costs.
    print(sum(costs))

if __name__ == "__main__":
    solve()