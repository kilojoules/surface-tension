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
    # Since we cannot use for-loops, we use a trick with a list comprehension
    # to populate the defaultdict.
    boxes = defaultdict(list)
    [boxes[A[i]].append(W[i]) for i in range(N)]
    
    # For each box, sort the weights of items inside it.
    # If a box has k items (k > 1), we keep the heaviest item and move the k-1 lightest.
    # We use a list comprehension to get the sorted weights for all boxes.
    # Then we slice [:-1] to get all but the heaviest item.
    sorted_weights_per_box = [sorted(boxes[i]) for i in range(1, N + 1)]
    
    # Extract the items that must be moved (all except the maximum weight in each box)
    # We use a nested comprehension and itertools.chain to flatten the list.
    items_to_move = chain.from_iterable(
        weights[:-1] for weights in sorted_weights_per_box if len(weights) > 1
    )
    
    # The total cost is the sum of the weights of all items that were moved.
    print(sum(items_to_move))

if __name__ == "__main__":
    solve()