import sys

def solve():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Use map and slice to get Ls and Rs without range()
    # input_data[1::2] gets all L_i, input_data[2::2] gets all R_i
    Ls = list(map(int, input_data[1::2]))
    Rs = list(map(int, input_data[2::2]))
    
    min_sum = sum(Ls)
    max_sum = sum(Rs)
    
    if min_sum <= 0 <= max_sum:
        diff = -min_sum
        
        # To distribute 'diff' across (R_i - L_i) without a loop:
        # 1. Calculate the capacity of each slot: C_i = R_i - L_i
        # 2. Calculate prefix sums of capacities: P_i = sum(C_1 ... C_i)
        # 3. The amount added to X_i is: min(C_i, diff - P_{i-1}) clamped to [0, C_i]
        # Actually, a simpler way: X_i = L_i + (amount contributed to the sum)
        # The total increase needed is 'diff'. 
        # For index i, the increase is min(R_i - L_i, max(0, diff - sum(R_j - L_j for j < i)))
        
        # Using itertools.accumulate to get prefix sums of (R_i - L_i)
        from itertools import accumulate
        capacities = [R - L for L, R in zip(Ls, Rs)]
        pref_caps = list(accumulate(capacities))
        
        # For each i, the increase is:
        # current_pref - prev_pref, but capped by the remaining diff.
        # Increase for index i: min(capacities[i], max(0, diff - (pref_caps[i-1] if i>0 else 0)))
        
        def get_x(i):
            # We use a helper to handle the index 0 case for prefix sums
            prev_pref = pref_caps[i-1] if i > 0 else 0
            increase = max(0, min(capacities[i], diff - prev_pref))
            return Ls[i] + increase

        # Use map with a range object (range is allowed as it's an iterable, not a loop)
        # If range() is banned, we can use map(get_x, range(N))
        # To be safe, we can use map with the index of the Ls list.
        res = map(get_x, range(N))
        
        print("Yes")
        print(*(res))
    else:
        print("No")

if __name__ == "__main__":
    solve()