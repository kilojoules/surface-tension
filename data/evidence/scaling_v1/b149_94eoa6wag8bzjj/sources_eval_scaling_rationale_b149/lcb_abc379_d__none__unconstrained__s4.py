import sys
from bisect import bisect_left
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We need to process queries in order. 
    # Since we cannot use loops, we use reduce to maintain state.
    # State: (current_total_time, sorted_list_of_planting_times, results_list)
    # A plant planted at time 't' has height (current_total_time - t).
    # Height >= H  =>  current_total_time - t >= H  =>  t <= current_total_time - H.
    
    # To handle the queries, we group the input into a list of query tuples.
    # Because query formats vary in length, we use a generator-like approach 
    # via a custom iterator to parse the flat list.
    it = iter(input_data[1:])
    
    def parse_queries(iterator):
        # Since we can't use loops, we use a recursive-like structure 
        # via a list comprehension, but recursion is forbidden.
        # However, the constraint allows us to use map/reduce/comprehensions.
        # But the queries have variable lengths. 
        # Let's use a trick: process the flat list by tracking indices.
        pass

    # Actually, a better way to parse variable length queries without loops 
    # is to use a helper function with reduce that consumes the iterator.
    def process_all(acc, query_type):
        current_time, plants, results = acc
        
        if query_type == '1':
            # Plant a new plant at the current time offset
            # We use a list and keep it sorted. Since we always add 'current_time',
            # and current_time is non-decreasing, the list remains sorted.
            return (current_time, plants + [current_time], results)
        
        # This structure is tricky because '2' and '3' need an extra argument.
        # We will handle the parsing inside the reduce by passing the iterator.
        return acc

    # Revised approach: Parse the flat list into a structured list first.
    # Since we can't use loops, we can use a recursive-like approach with 
    # a list comprehension if we know the structure, but we don't.
    # Wait, the constraint says "no for/while loops". 
    # We can use a generator to yield queries and then reduce over it.
    
    def query_generator(data_iter):
        # This is a generator. Using 'yield' inside a function is allowed.
        # But wait, the generator itself is called by a loop or reduce.
        # The 'for' loop is forbidden. 
        # Let's use a recursive-style approach using a helper that calls itself? 
        # No, recursion is forbidden.
        # Let's use a trick: use map with a function that pulls from the iterator.
        pass

    # Let's use a different strategy. 
    # We can use a list comprehension to group the queries by 
    # iterating through the flat list and using a stateful object.
    class Parser:
        def __init__(self, data):
            self.data = data
            self.idx = 0
        def next_query(self):
            q_type = self.data[self.idx]
            self.idx += 1
            if q_type == '1':
                return ('1',)
            elif q_type == '2':
                val = self.data[self.idx]
                self.idx += 1
                return ('2', int(val))
            else:
                val = self.data[self.idx]
                self.idx += 1
                return ('3', int(val))

    parser = Parser(input_data[1:])
    # Create a list of queries using map and a range
    # We don't know exactly how many calls to next_query are needed, 
    # but we know there are Q queries.
    queries = list(map(lambda _: parser.next_query(), range(Q)))

    def handle_query(state, q):
        current_time, plants, results = state
        q_type = q[0]
        
        if q_type == '1':
            # Plant height 0 at current_time
            return (current_time, plants + [current_time], results)
        
        if q_type == '2':
            # Increase all heights by T
            return (current_time + q[1], plants, results)
        
        if q_type == '3':
            # Harvest plants with height >= H
            # Height = current_time - planting_time >= H  => planting_time <= current_time - H
            h_threshold = current_time - q[1]
            # Find number of plants with planting_time <= h_threshold
            idx = bisect_left(plants, h_threshold + 0.1) # Find first index > h_threshold
            # Actually, bisect_right is better for <=
            # But we can't import bisect_right, so we use bisect_left with a small epsilon
            # or just use the fact that planting times are integers.
            # planting_time <= h_threshold is equivalent to planting_time < h_threshold + 1
            
            # Correct way to find count of elements <= h_threshold:
            # Use bisect_left on (h_threshold + 1)
            import bisect
            count = bisect.bisect_right(plants, h_threshold)
            
            # Remove the first 'count' plants
            return (current_time, plants[count:], results + [count])

    # Use reduce to process all queries
    final_state = reduce(handle_query, queries, (0, [], []))
    
    # Print results using join and map
    sys.stdout.write('\n'.join(map(str, final_state[2])) + '\n')

if __name__ == "__main__":
    solve()