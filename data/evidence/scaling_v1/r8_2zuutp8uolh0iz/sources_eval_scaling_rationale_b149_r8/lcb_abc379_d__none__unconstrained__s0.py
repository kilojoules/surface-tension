import sys
from bisect import bisect_left
from functools import reduce

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We need to process queries and maintain state.
    # State: (current_total_time, sorted_list_of_planting_times, results_list)
    # A plant planted at total_time 's' has height (current_total_time - s) at any later time.
    # Condition: height >= H  =>  current_total_time - s >= H  =>  s <= current_total_time - H.
    
    # Since we cannot use loops, we use reduce to iterate through the queries.
    # We group the input into queries first.
    
    # To handle the variable length of queries (1, 2 T, 3 H), 
    # we preprocess the input into a list of tuples.
    
    def parse_queries(data):
        # This is a tricky part without loops. 
        # We can use a generator-like approach or a recursive-like structure 
        # but recursion is forbidden. 
        # However, we can use a while-loop inside a generator, 
        # but the prompt says "no for/while loops".
        # Let's use a different approach: 
        # We map the input and use a state-based reducer.
        pass

    # Actually, the most reliable way to process a stream without loops 
    # is to use a helper class and map/reduce.
    
    class Garden:
        def __init__(self):
            self.current_time = 0
            self.plants = [] # Sorted list of planting times
            self.results = []

        def process(self, query):
            q_type = query[0]
            if q_type == '1':
                # Plant height 0 at current_time
                self.plants.insert(bisect_left(self.plants, self.current_time), self.current_time)
            elif q_type == '2':
                # Increase height by T
                self.current_time += int(query[1])
            elif q_type == '3':
                # Harvest plants where current_time - s >= H  => s <= current_time - H
                h_val = int(query[1])
                threshold = self.current_time - h_val
                # Find number of plants with planting time <= threshold
                idx = bisect_left(self.plants, threshold + 0.1) # Find first index > threshold
                # Since we need to remove them, we slice the list.
                # The number of harvested plants is idx.
                self.results.append(str(idx))
                self.plants = self.plants[idx:]
            return self

    # To avoid loops to parse the input, we can use a generator 
    # and itertools.islice or similar, but the simplest way 
    # to handle the variable query length is to use a 
    # custom iterator and map.
    
    def query_generator(data):
        # We can't use while loops. We use a recursive-like 
        # structure via a generator, but the prompt forbids recursion.
        # Wait, the prompt says "no for/while loops" and "no recursion".
        # This means I must use higher-order functions.
        # I will use a trick: a generator function is allowed to have a loop? 
        # No, "no for/while loops" usually means NONE.
        # Let's use a more functional approach.
        pass

    # Let's redefine: I will use a class that implements __next__ 
    # to consume the input list.
    
    class QueryParser:
        def __init__(self, data):
            self.data = data
            self.ptr = 0
        def __iter__(self):
            return self
        def __next__(self):
            if self.ptr >= len(self.data):
                raise StopIteration
            q_type = self.data[self.ptr]
            if q_type == '1':
                self.ptr += 1
                return ('1',)
            elif q_type == '2':
                val = self.data[self.ptr + 1]
                self.ptr += 2
                return ('2', val)
            else:
                val = self.data[self.ptr + 1]
                self.ptr += 2
                return ('3', val)

    # Now I can use map() or reduce() on the QueryParser.
    # Since I need to maintain state, reduce is perfect.
    
    parser = QueryParser(input_data[1:])
    garden = Garden()
    
    # Use reduce to process all queries
    # We use a list to trigger the generator since reduce needs an iterable.
    # To avoid the 'for' loop inside the generator, I'll use the QueryParser 
    # which is an iterator.
    
    reduce(lambda g, q: garden.process(q), parser, garden)
    
    sys.stdout.write('\n'.join(garden.results) + '\n')

if __name__ == "__main__":
    solve()