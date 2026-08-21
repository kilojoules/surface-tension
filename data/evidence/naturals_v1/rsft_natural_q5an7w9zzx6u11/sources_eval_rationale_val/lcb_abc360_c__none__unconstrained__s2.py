import sys
from collections import defaultdict
from itertools import chain

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    A = list(map(int, input_data[1:N+1]))
    W = list(map(int, input_data[N+1:]))
    
    # Group weights by box using a dictionary
    # We use a list comprehension to populate the defaultdict
    # Note: we use a trick with a dummy list to execute the grouping in one line
    boxes = defaultdict(list)
    [boxes[box].append(weight) for box, weight in zip(A, W)]
    
    # For each box, if it has k items (k > 1), we must move k-1 items.
    # To minimize cost, we keep the heaviest item and move the k-1 lightest.
    # 1. Sort weights in each box.
    # 2. For boxes with k > 1, take the first k-1 elements.
    
    # We use a list comprehension to get the sorted weights for every box
    # and then another to slice the 'to-be-moved' items.
    sorted_weights = [sorted(boxes[i]) for i in range(1, N + 1)]
    
    # For each list of weights, if length is k, we move the first k-1 items.
    # The slice [:len(lst)-1] handles both k=1 (empty list) and k>1.
    to_move = [lst[:len(lst)-1] for lst in sorted_weights]
    
    # Flatten the list of lists and sum the weights
    total_cost = sum(chain.from_iterable(to_move))
    
    print(total_cost)

if __name__ == "__main__":
    solve()