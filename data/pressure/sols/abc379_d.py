import sys
import numpy as np


def main():
    data = sys.stdin.read().split('\n')
    q = int(data[0])
    lines = np.array(data[1:1 + q], dtype='U')
    parts = np.char.partition(np.char.strip(lines), ' ')
    types = parts[:, 0].astype(np.int64)
    vals_str = parts[:, 2]
    vals = np.where(vals_str == '', '0', vals_str).astype(np.int64)

    t_add = np.where(types == 2, vals, 0)
    cse = np.cumsum(t_add) - t_add  # cumulative T before each query

    births = cse[types == 1]  # non-decreasing already

    mask3 = types == 3
    th = cse[mask3] - vals[mask3]
    s = np.searchsorted(births, th, side='right')
    m = np.maximum.accumulate(s)
    ans = np.diff(m, prepend=0)

    sys.stdout.write('\n'.join(ans.astype('U').tolist()))


main()
