from drbg.hmac_drbg import Hmac_drbg
from shawe_taylor.main import random_prime_routine
from tools.states import FAILURE, SUCCESS
from tools.gen_use_funs import mult_inv, sha3_256_int, bits_to_int, int_to_bytes
from math import ceil, gcd, isqrt

# Valid nlen values according to FIPS 186-5 Section 5.1 (typical sizes)
VALID_NLEN = {1024, 2048, 3072, 4096}

'''
 4096-bit shows up in VALID_NLEN even though it's absent from your security map
 because it's a real-world “sweet spot” between 3072 and 7680 bits.

 SECURITY_STRENGTH_MAP is the theoretical mapping.

 VALID_NLEN is the practical subset commonly implemented, chosen for a balance
 of security, speed, and compatibility.
'''
#TODO clear of chat comments
# Mapping from nlen to security_strength in bits (based on SP 800-57 Part 1)
SECURITY_STRENGTH_MAP = {
    1024: 80,
    2048: 112,
    3072: 128,
    4096: 192, # nlen = 4096 is not to be found in NIST SP 800-57 PART 1 REV. 5
               # security strength table, but this value falls between 3072 and 7680 thus 192 used (4096 exeeds 3072)
    7680: 192,
    15360: 256
}

def get_seed(nlen: int):
    """
    Implementation based on FISP 186-5: A 1.2.1
    Function needed to generate seed for generate_provable_prime_pair function

    hmac_drbg_generate_function() returns bitstring, so this function
    also returns seed_bits as bitstring
    """
    if nlen not in VALID_NLEN:
        return FAILURE, 0

    security_strength = SECURITY_STRENGTH_MAP.get(nlen)
    if security_strength is None:
        return FAILURE, 0

    seed_bits_length = 2 * security_strength 
    hmac_drbg = Hmac_drbg(requested_instantiation_security_strength=security_strength)
    status, seed_bits = hmac_drbg.hmac_drbg_generate_function(seed_bits_length) 

    return SUCCESS, seed_bits

def generate_provable_prime_pair(nlen, e, seed):
    """
    Implementation based on FISP 186-5: A 1.2.2
    """
    if nlen < 2048:
        return FAILURE, 0, 0
    if e <= 65536 or e >= (1 << 256) or e % 2 == 0: #If ((e ≤ 2^16) OR (e ≥ 2^256) OR (e is not odd))
        return FAILURE, 0, 0
    security_strength = SECURITY_STRENGTH_MAP.get(nlen)
    if len(seed) < 2 * security_strength:
        return FAILURE, 0, 0
    working_seed = seed

    status, p, p1, p2, pseed = construct_provable_prime(l=nlen >> 1, n1=1, n2=1, firstSeed=bits_to_int(working_seed), e=e) #nlen >> 1 = nlen/2
    if not status:
        return FAILURE, 0, 0
    working_seed = pseed
    while True:
        status, q, q1, q2, qseed = construct_provable_prime(l=nlen >> 1, n1=1, n2=1, firstSeed=working_seed, e=e)
        if not status:
            return FAILURE, 0, 0
        working_seed = qseed
        if not abs(p - q) <= (1 << ((nlen >> 1)  - 100)):
            break
    pseed = 0
    qseed = 0
    working_seed = 0
    return SUCCESS, p, q

def construct_provable_prime(l: int, n1, n2, firstSeed: int, e):
    """
    Implementation based on FISP 186-5: B.10
     1. L A positive integer equal to the requested bit-length for p. Note that acceptable values for 
       L = nlen/2 are computed as specified in Appendix B.3.1, criteria 2(b) and (c), with nlen assuming 
       a value specified in Table B.1.
     2. N1 A positive integer equal to the requested bit-length for p1. If N1 ≥ 2, then p1 is an odd prime
       of N1 bits; otherwise, p1 = 1. Acceptable values for N1 ≥ 2 are provided in Table A.1
     3. N2 A positive integer equal to the requested bit-length for p2. If N2 ≥ 2, then p2 is an odd prime
       of N2 bits; otherwise, p2 = 1. Acceptable values for N2 ≥ 2 are provided in Table A.1
     4. firstseed (not) a bit string (in this implementation bitstring is converted to int before passing here,
       so firstseed will be int) equal to the first seed to be used.
     5. e The public verification exponent.
    """
    hashlen = 256
    p1 = None
    p2seed = None
    p2 = None
    p0seed = None
    p0 = None
    pseed = None
    # If L, N1, and N2 are not acceptable, then return (FAILURE, 0, 0, 0, 0). TODO What does it mean "not acceptable"
    nlen = 2*l
    # SECURITY_STRENGTH_MAP contains nlen as keys, so if nlen = l * 2 it's sufficient to check nlen in SECURITY_STRENGTH_MAP
    if not nlen in SECURITY_STRENGTH_MAP \
    or not auxilary_prime_len_acceptable(n1, nlen) \
    or not auxilary_prime_len_acceptable(n2, nlen):
        return FAILURE, 0, 0, 0, 0

    if n1 == 1:
        p1 = 1
        p2seed = firstSeed
    if n1 >= 2:
        st_data = random_prime_routine(n1, firstSeed)
        p1 = st_data.prime
        p2seed = st_data.prime_seed
        if st_data.status == FAILURE:
            return FAILURE, 0 ,0 ,0 ,0
    if n2 == 1:
        p2 = 1
        p0seed = p2seed
    if n2 >= 2:
        st_data = random_prime_routine(n2, p2seed)
        p2 = st_data.prime
        p0seed = st_data.prime_seed
        if st_data.status == FAILURE:
            return FAILURE, 0 ,0 ,0 ,0
        
    st_data = random_prime_routine(ceil(l / 2) + 1, p0seed)
    # print("status = {0}, prime = {1}, prime_seed = {2}, prime_gen_counter = {3}".format(st_data.status, st_data.prime, st_data.prime_seed, st_data.prime_gen_counter))
    p0 = st_data.prime
    pseed = st_data.prime_seed
    if st_data.status == FAILURE:
        print("random_prime_routine failed")
        return FAILURE, 0 ,0 ,0 ,0
    
    if gcd(p0 * p1, p2) != 1:
        print("gcd(p0 * p1, p2) != 1 FAILURE")
        return FAILURE, 0 ,0 ,0 ,0
    num_hash_blocks = ceil(l / hashlen) - 1
    pgen_counter = 0
    x = 0
    for i in range(num_hash_blocks):
        x += sha3_256_int(int_to_bytes(pseed + i)) * (1 << (i * hashlen)) # 1 << x is the same as pow(2, x)
    pseed += num_hash_blocks + 1
    x = isqrt(1 << (2*l - 1)) + (x % ((1 << l) - isqrt(1 << (2*l - 1)))) # isqrt(...) is the same as int(sqrt(2)*pow(2,l-1)) but ensures no precission loss
    y = mult_inv(p0 * p1, p2)
    if y == None:
        print("y == NONE FAILURE")
        return FAILURE, 0 ,0 ,0 ,0 
    t = ceil(((2 * y * p0 * p1) + x) / (2 * p0 * p1 * p2))
    while True:
        if (2*(p2-y) * p0 * p1 + 1) > (1 << l):
            t = ceil((2 * y * p0 * p1 + isqrt(1 << (2*l - 1))) / (2 * p0 * p1 * p2))
        p = 2 * (t * p2 - y) * p0 * p1 + 1
        pgen_counter += 1

        if gcd(p - 1, e) == 1:
            a = 0
            for i in range(num_hash_blocks):
                a = a + sha3_256_int(int_to_bytes(pseed + i)) * (1 << (i * hashlen))
            pseed += num_hash_blocks + 1
            a = 2 + (a % (p - 3))
            z = pow(a, 2 * (t * p2 - y) * p1, p)
            if gcd(z - 1, p) == 1 and 1 == pow(z, p0, p):
                return SUCCESS, p, p1, p2, pseed
        if pgen_counter >= 5 * l:
            # print()
            return FAILURE, 0 ,0 ,0 ,0
        t += 1  


def auxilary_prime_len_acceptable(n, l):
    """
    n - auxilary prime size
    l - nlen
    Acceptable auxiliary prime sizes according to FIPS 186-5, Table A.1.
    If n == 1, it indicates no auxiliary prime (which is acceptable).
    Check if auxilary primes sizes are acceptabe - if the gap between auxilary primes
    and desired prime size isn't too big:
    "The limiting factor is speed — the bigger the gap between your auxiliaries
    and your target prime size, the longer it takes to find a candidate that passes the proof conditions."
    """
    if n == 1:
        return True
    if 2048 <= l <= 3071 and n > 140:
        return True
    if 3072 <= l <= 4095 and n > 170:
        return True
    if 4096 <= l and n > 200:
        return True
    return False

# print(get_seed(1024))
# state, p, p1, p2, pseed = construct_provable_prime(1024, 1, 1, firstSeed=312467831246783124678, e=65537)
# print("state = {0}, p = {1}, p1 = {2}, p2 = {3}, pseed = {4}".format(state, p, p1, p2, pseed))