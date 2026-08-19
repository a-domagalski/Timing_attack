from secrets import token_bytes
from scipy.stats import entropy
from collections import Counter
from cryptography.hazmat.primitives import hashes
# from cryptography.hazmat.primitives.kdf.hkdf import HKDF
import numpy as np
import math
from tools.gen_use_funs import bytes_to_bits
from tools.states import FAILURE, SUCCESS
# zmodyfikowane wzgledem standardu NIST (800-90C str.33) - entropia nie jest dodawana
# a liczona w całości po konkatenacji bitstringów, dlatego funkcja GetEntropy zastąpoina
# jest funkcją get_entropy_bitstring i nie zwraca entropii 

def shannon_entropy_scipy(bitstring):
    # print(bitstring)
    counts = Counter(bitstring)
    # print(counts)
    total = len(bitstring)
    # print("bitstring len in shannons entropy fun ", total)
    probs = np.array([count / total for count in counts.values()])
    # print(probs)
    return entropy(probs, base=2)

# Can't get physical entropy thus secrets module used instead (won't be 
# tully secure, but i think it's enough for it's job)
# Returns bytes, so need to divide len by 8 
def get_entropy_bitstring():
    random_bytes = token_bytes(4)
    # print("random bytes: ", random_bytes)
    bitstring = bytes_to_bits(random_bytes)
    # print("bytes_to_bits: ", bitstring)
    return SUCCESS, bitstring

# Input: integer (min_entropy, min_length, max_length, prediction_resistance_request).
# Output: string status, bitstring returned_bits.
# From NIST SP 800 A1 p. 38:
# Minimum entropy input length (min_length) = security_strength
# Maximum entropy input length (max_length) = 2^35 bits = 3459738368
def get_entropy_input(min_entropy, min_lenght, max_length=3459738368, prediction_resistance_request=None):
    # print("executing get_entropy_input()")
    if min_lenght > max_length:
        return FAILURE, None
    tmp = ''
    entropy_total = 0
    x = 0
    while entropy_total < min_entropy:
        status, entropy_bitstring = get_entropy_bitstring()
        if status == FAILURE:
            return status, None
        tmp = tmp + entropy_bitstring
        entropy_total += shannon_entropy_scipy(tmp) # 4.4 entropy_total = entropy_total + assessed_entropy
        # print("min_entropy={0}, entropy_total={1}".format(min_entropy, entropy_total))
        x += 1
        # print(x)
    # print("entropy_total: ", entropy_total)
    bitstring_len = len(tmp)
    # print("bitstring_len ", bitstring_len)
    if bitstring_len < min_lenght:
        tmp = tmp + '0' * (min_lenght - bitstring_len)
        # print("bitstring len after padding added ", len(tmp))
    if max_length != None and bitstring_len > max_length:
        # print("Bitstring in tmp ", tmp)
        bit_int = int(tmp, 2)
        # print("bit_int:: ", bit_int)

        # print("Bitstring len: ", len(tmp))

        derived_val = hash_derivation_fun(tmp, max_length)
        # print("derived_val: ", derived_val)
        # print("derived_val to bits: ", bytes_to_bits(derived_val))
        # print("entropy of derived: ", shannon_entropy_scipy(derived_val))
        # print("Derived bitstring len: ", len(derived_val))
    # print("end of get_entropy_input()")
    return SUCCESS, tmp

# def bitstring_to_bytes(bitstring):
#     print("Converting bitstring: {}".format(bitstring) + " to bytes")
#     return int(bitstring, 2).to_bytes()

def bitstring_to_bytes(bitstring: str) -> bytes:
    # Pad with zeros if not divisible by 8
    padding = (8 - len(bitstring) % 8) % 8
    bitstring_padded = bitstring + '0' * padding
    return int(bitstring_padded, 2).to_bytes(len(bitstring_padded) // 8, byteorder='big')

# bitstring: The string to be hashed
# no_of_bits: The number of bits to be returned by Hash_df.
#   The maximum length (max_number_of_bits) is implementation dependent, 
#   but shall be less than or equal to (255 × outlen). no_of_bits_to_return 
#   is represented as a 32-bit integer.
def hash_derivation_fun(bitstring, no_of_bits):
    temp = ''
    len = math.ceil(no_of_bits/256)
    counter = 1
    counter = counter.to_bytes(1, byteorder='big')
    for i in range(len):
        # print("bytes_to_bits(counter) ", bytes_to_bits(counter))
        tmp = int(no_of_bits).to_bytes(4, byteorder='big')
        # print("bytes_to_bits(tmp), ", bytes_to_bits(tmp))
        tmp_bitstr = bytes_to_bits(counter) + bytes_to_bits(tmp) + bitstring
        digest = hashes.Hash(hashes.SHA256())
        digest.update(bitstring_to_bytes(tmp_bitstr))
        temp = temp + bytes_to_bits(digest.finalize())
        counter += int(1).to_bytes(1, byteorder='big')
    requested_bits = temp[0:no_of_bits]
    return requested_bits
# status, data = get_entropy()
# bitstring = bytes_to_bits(data)
# print("bitstring: {} \nof len: {}".format(bitstring, len(bitstring)))  # Output: '1001001011111001'

# =========================================================================================
# testing
# get_entropy_input(0.999, min_lenght=100, max_length=101) 

# bstr = '11100'
# print(shannon_entropy_scipy(bstr))