from sympy import isprime
from provable_prime import construct_provable_prime
from random import randint
import time
from statistics import mean

def test_construct_provable_prime():
    time_table = []
    overall_time = 0
    for i in range(0, 100):
        rnd = randint(0, int(1e20))
        # print("input_seed: ", rnd)
        t1 = time.time()
        state, p, p1, p2, pseed = construct_provable_prime(2048, 1, 1, rnd, 65537)
        t2 = time.time()
        t_diff = t2 - t1
        overall_time += t_diff
        time_table.append(t_diff)
        print("Iteration: {0}, exec. time: {1}, current mean time: {2}".format(i, t_diff, mean(time_table)))

        if not state:
            return False
        elif not isprime(p):
            return False
    print("mean exec time: ", mean(time_table))
    print("overall_time: ", overall_time)
    return True

print(test_construct_provable_prime())