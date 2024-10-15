import math
import argparse

def legendre(p, n):
    return sum([math.floor(n / p**i) for i in range(1, math.floor(math.log(n, p)) + 1)])

def max_legendre(p, upper_bound):
    curr = 1
    N = 1
    while curr < upper_bound:
        N += 1
        curr = legendre(p, N)
    return N - 1

def zero_ideal_size(p, w):
    N = max_legendre(p, w)
    return sum([legendre(p, n) for n in range(1, N + 1)])

def num_unique_polys(p, w):
    N = max_legendre(p, w)
    print(N)
    return w*(N+1) - zero_ideal_size(p, w)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--legendre", type=int, nargs=2)
    parser.add_argument("--zero-ideal-size", type=int, nargs=2)
    parser.add_argument("--num-unique-polys", type=int, nargs=2)
    parser.add_argument("--max-legendre", type=int, nargs=2)
    args = parser.parse_args()
    if args.legendre:
        print(legendre(*args.legendre))
    if args.zero_ideal_size:
        print(f"{args.zero_ideal_size[0]} ^ {zero_ideal_size(*args.zero_ideal_size)}")
    if args.num_unique_polys:
        p, w = args.num_unique_polys
        print(f"{p} ^ {num_unique_polys(p, w)}")
    if args.max_legendre:
        print(max_legendre(*args.max_legendre))