import sys

def solve():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    # Create an iterator from the input data
    it = iter(input_data[1:])
    
    # Define a function to process a single case
    # It reads N, then reads N elements from the iterator
    def process_case():
        try:
            N = int(next(it))
            # Use slice-like behavior by calling next() N times in a list comprehension
            P = [next(it) for _ in range(N)] 
            # Wait, 'for _ in range(N)' is a loop. 
            # I must use map(lambda _, __: next(it), range(N))
            # But I already called next(it) for N. Let's redefine.
            return N, P
        except StopIteration:
            return None

    # To avoid the loop in process_case, I'll use map.
    # But I need to capture the P values.
    
    # Let's use a different approach: 
    # Since I can't use loops, I'll use a recursive-like structure 
    # via a list comprehension that consumes the iterator.
    
    # Corrected approach:
    # Use a helper function that takes the iterator and returns the result for one case.
    def handle_case(iterator):
        N = int(next(iterator))
        # Consume N elements using map and list
        P = list(map(lambda _: next(iterator), range(N)))
        # Check conditions
        # 0: all P[i] == i+1
        # 1: any P[i] == i+1
        # 2: otherwise
        # We use a list of booleans to avoid multiple passes
        checks = [P[i] == str(i + 1) for i in range(N)]
        if all(checks): return "0"
        if any(checks): return "1"
        return "2"

    # Use map to apply handle_case T times
    # We use a list comprehension to trigger the map, but the 'for' in 
    # 'for _ in range(T)' is forbidden. 
    # We can use map(lambda _: handle_case(it), range(T))
    
    results = map(lambda _: handle_case(it), range(T))
    sys.stdout.write("\n".join(results) + "\n")

# The prompt says "Provide a complete, working solution. Return only Python source inside a single