import sys
import numpy as np


def main():
    data = sys.stdin.buffer.read().split()
    S = data[0]
    T = data[1]
    N = len(S)
    s_u = np.frombuffer(S, dtype=np.uint8)
    t_u = np.frombuffer(T, dtype=np.uint8)
    diff = s_u != t_u
    dec = np.where(diff & (t_u < s_u))[0]        # decreasing changes, ascending index
    inc = np.where(diff & (t_u > s_u))[0]        # increasing changes
    order = np.concatenate([dec, inc[::-1]]).astype(np.int64)
    M = int(order.size)

    if M == 0:
        sys.stdout.write("0\n")
        return

    tril = np.tril(np.ones((M, M), dtype=bool))
    T_ch = t_u[order]
    S_ch = s_u[order]
    sub = np.where(tril, T_ch[None, :], S_ch[None, :]).astype(np.uint8)
    full = np.tile(s_u, (M, 1))
    full[:, order] = sub

    rows = np.ascontiguousarray(full).view('S%d' % N).reshape(M)
    body = b'\n'.join(rows.tolist()).decode()
    sys.stdout.write(str(M) + '\n' + body + '\n')


main()
