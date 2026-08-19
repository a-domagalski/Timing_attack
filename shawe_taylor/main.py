# This file contains implementation of Shawe-Taylor prime number 
# generation algorithm.
# - Hash function used: sha3-256
# - Primality test: Miller-Rabin Probabilistic primality test

from sympy import primerange
from math import floor
from math import ceil
from math import isqrt
from math import gcd
from .shawe_taylor_d import ShaweTaylorD
from tools.states import FAILURE, SUCCESS
from tools.gen_use_funs import sha3_256_int, int_to_bytes, bits_to_int
from drbg.hmac_drbg import Hmac_drbg
import copy # for testing purposes - to be deleted

_SMALL_PRIMES = list(primerange(2, 10000))

# Doesn't return bytes but integers
def random_prime_routine(length: int, input_seed: int):
    """
    length: int - The length of the prime to be generated.
    input_seed: int - The seed to be used for the generation of the requested prime.
    """
    hashlen = 256    
    if length < 2:
        return ShaweTaylorD(FAILURE)
    if length < 33:
        prime_seed = input_seed
        prime_gen_counter = 0
        while True:
            # Generate a pseudorandom integer c of length bits.
            c = sha3_256_int(int_to_bytes(prime_seed)) ^ sha3_256_int(int_to_bytes(prime_seed + 1))
            c = pow(2, length - 1) + (c % (1 << length - 1))
            # print(c.bit_length())
            # print(c) # for testing purposes - to be deleted
            # tmp_c = copy.deepcopy(c) # for testing purposes - to be deleted
            # print(float(c))  # You'll get something like 8.988e+307 (only 15 significant digits)
            # print(floor(c / 2))  # This is floor(approximation), not exact c//2
            # print(c // 2)        # Exact
            # print((c // 2) == floor(c / 2))  # False for large c
            # print((c // 2) == (c >> 1))
            '''
            10.05.2025: przemyśleć lepszy (szybszy) sposób dzielenia (np. przesunięcie bitów w prawo
            (chociaż to problematyczne bo nie zachowuje części ułamkowej))
            edit 13.10.2025 - c / 2 powoduje konwersję na float. Przy dużych liczbach, przekraczających 
            64 bity (float jest 64 bitowy) float będzie się zaokrąglał tracąc prawdziwą wartość.
            Dlatego zamienione z:
            c = (2 * floor(c / 2)) + 1
            '''
            c = (2 * floor(c >> 1)) + 1
            
            # print(type(c)) # for testing purposes - to be deleted
            # print(c) # for testing purposes - to be deleted
            # temp = (2 * (tmp_c >> 1)) + 1 # for testing purposes - to be deleted
            # print(temp) # for testing purposes - to be deleted
            
            # Set prime to the least odd integer greater than or equal to c.
            prime_gen_counter += 1
            prime_seed += 2
            prime = None
            if trial_division_extended(c):
                prime = c
                return ShaweTaylorD(status=SUCCESS, prime=prime, prime_seed=prime_seed, prime_gen_counter=prime_gen_counter)
            if prime_gen_counter > 4 * length:
                return ShaweTaylorD(FAILURE)
    # temp_len = length / 2 + 1
    # temp_len
    '''
    problematic usage of st_data here (see assigning below) 
    doesnt make sense to use st_data. Maybe worth replacing 
    usage of st_data.prime with just prime and eliminate usage of st_data at all
    ''' 
    st_data = random_prime_routine (( ceil(length / 2) + 1), input_seed)                                       
    prime_seed = st_data.prime_seed
    prime_gen_counter = st_data.prime_gen_counter
    if st_data.status == FAILURE:
        return ShaweTaylorD(FAILURE)
    iterations = ceil(length / hashlen) - 1
    old_counter = prime_gen_counter

    # Generate a pseudorandom integer x in the interval [2length – 1, 2length].
    x = 0
    for i in range (iterations):
        hash = sha3_256_int(int_to_bytes(prime_seed + i))
        x = x + (hash * pow(2, i * hashlen))
    prime_seed = prime_seed + iterations + 1
    x = pow(2, length - 1) + x % pow(2, length - 1)
    # Generate a candidate prime c in the interval [2length – 1, 2length].
    t = ceil(x / (2 * st_data.prime))
    temp = st_data.prime
    temp
    while True:
        t
        temp = 2 * t * st_data.prime + 1
        temp
        if 2 * t * st_data.prime + 1 > (1 << length):
            t = ceil(pow(2, length - 1) / (2 * st_data.prime))
            t
        c = 2 * t * st_data.prime + 1
        prime_gen_counter += 1
        # The remaining steps test the candidate prime c for primality.
        a = 0
        for i in range (iterations):
            hash = sha3_256_int(int_to_bytes(prime_seed + i))
            a = a + (hash * pow(2, i * hashlen))
        prime_seed = prime_seed + iterations + 1
        a = 2 + (a % (c - 3))
        # print("a: {0}, t: {1}, c: {2}".format(a, t, c))
        z = pow(a, 2 * t, c)
        if 1 == gcd(z - 1, c) and 1 == pow(z, st_data.prime, c):
            prime = c
            return ShaweTaylorD(SUCCESS, prime, prime_seed, prime_gen_counter)
        if prime_gen_counter >= (4 * length + old_counter):
            return ShaweTaylorD(FAILURE)
        t += 1

#test number for primality
def trial_division_extended(prime_to_test):
    """
    Extended trial division means that if no divisors were found in _SMALL_PRIMES
    it doesn't mean that prime_to_test is in fact prime (it wasn't tested for all 
    required range of primes that is primerange(2, isqrt(prime) + 1)), but becouse 
    trial_division is suitable only for small numbers we need to run
    miller_rabin_primality_test good for large numbers
    """
    if prime_to_test < 2:
        return False
    for p in _SMALL_PRIMES:
        if prime_to_test % p == 0:
            return prime_to_test == p  # True only if prime_to_test equals that small prime
    return miller_rabin_primality_test(prime_to_test)

#sympy implements sieve of Eratosthenes in primerange
def sqrt_limit_primerange(prime):
    # print("isqrt(prime) + 1: ", isqrt(prime) + 1)
    return primerange(2, isqrt(prime) + 1)

def miller_rabin_primality_test(w: int) -> bool:
    """
    Miller–Rabin probabilistic primality test per FIPS 186-5 B.3.1,
    using a DRBG interface compatible with HMAC-DRBG:
        state, bits = hmac_drbg_generate_function(requested_bits)
    where:
        - state: True on success, False on failure
        - bits: pseudorandom bitstring ('0'/'1' string or bytes)

    Returns:
        True - if PROBABLY PRIME
        False - if COMPOSITE
    """
    hmac_drbg = Hmac_drbg(requested_instantiation_security_strength=256) 
    # Validate input
    if w < 2:
        return False
    if w in (2, 3):
        return True
    if w % 2 == 0:
        return False
    # Step 1. Find a such that 2^a divides w−1
    d = w - 1
    a = 0
    temp = d
    while temp % 2 == 0:
        a += 1
        temp //= 2
    # Step 2. m = (w−1) / 2^a
    m = temp
    # Step 3. wlen = len(w)
    wlen = w.bit_length()

    # Step 4. Iterations
    for i in range(fips186_5_recommended_iterations(wlen)):
        # 4.1 Obtain pseudorandom bits from DRBG until valid
        max_attempts = 1000
        b_int = None
        for attempt in range(max_attempts):
            state, pseudo_bits = hmac_drbg.hmac_drbg_generate_function(wlen)
            if not state:
                raise RuntimeError("HMAC-DRBG generate failed")

            b = bits_to_int(pseudo_bits)
            if 1 < b < w - 1:
                b_int = b
                break
        if b_int is None:
            # fallback: map pseudorandom bits to valid range (2, w−2)
            state, pseudo_bits = hmac_drbg.hmac_drbg_generate_function(wlen)
            if state == FAILURE:
                raise RuntimeError("HMAC-DRBG generate failed")
            b = bits_to_int(pseudo_bits)
            b_int = (b % (w - 3)) + 2

        # 4.3 z = b^m mod w
        z = pow(b_int, m, w)

        # 4.4 check if z == 1 or w−1
        if z == 1 or z == w - 1:
            continue

        # 4.5 loop j = 1 to a−1
        composite = True
        for j in range(1, a):
            z = (z * z) % w
            if z == w - 1:
                composite = False
                break
            if z == 1:
                return False
        if composite:
            return False

    # Step 5
    return True

def fips186_5_recommended_iterations(bit_length: int) -> int:
    if bit_length >= 8192:
        return 1
    elif bit_length >= 6144:
        return 2
    elif bit_length >= 5120:
        return 2
    elif bit_length >= 4096:
        return 3
    elif bit_length >= 3072:
        return 3
    elif bit_length >= 2048:
        return 4
    elif bit_length >= 1536:
        return 3
    elif bit_length >= 1024:
        return 4
    elif bit_length >= 768:
        return 5
    elif bit_length >= 512:
        return 7
    else:
        return 10

# print(trial_division(126431802))

# shd = random_prime_routine(1024, 1234)
# print("status = {0}, prime = {1}, prime_seed = {2}, prime_gen_counter = {3}".format(shd.status, shd.prime, shd.prime_seed, shd.prime_gen_counter))


