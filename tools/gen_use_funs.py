from tools.states import FAILURE, SUCCESS
from hashlib import sha3_256

def hex_to_bitstring(hex_str: str) -> str:
    # Remove '0x' prefix if present
    if hex_str.startswith("0x") or hex_str.startswith("0X"):
        hex_str = hex_str[2:]
    
    # Convert hex to integer
    num = int(hex_str, 16)
    
    # Convert integer to bit string without '0b' prefix
    bit_str = bin(num)[2:]
    
    # Pad the bit string with leading zeros to make length a multiple of 4
    bit_str = bit_str.zfill(len(hex_str) * 4)
    
    return bit_str

def bytes_to_bits(byte_data: bytes) -> str:
    return ''.join(f'{byte:08b}' for byte in byte_data)

def bits_to_bytes(bits) -> bytes:
    if bits == None:
        return b""
    else:
        return bytes(int(bits[i:i+8], 2) for i in range(0, len(bits), 8))
def int_to_bytes(num: int) -> bytes:
    """
    Helper function converting int to bytes
    """
    return num.to_bytes((num.bit_length() + 7) // 8 or 1, 'big')

def bits_to_int(bitstring: str) -> int:
    return int(bitstring, 2)
    # return int.from_bytes(bits_to_bytes(bitstring))

def mult_inv(z: int, a: int):
    """
    x * z mod a = 1 - find x
    """
    if not (a >= 0 and z >= 0 and z < a):
        return FAILURE
    i = a
    j = z
    y2 = 0
    y1 = 1
    while j > 0:
        quotient = i // j
        remainder = i - ( j * quotient)
        y = y2 - (y1 * quotient)
        i = j
        j = remainder
        y2 = y1
        y1 = y
    if (i != 1):
        return FAILURE
    else:
        return SUCCESS, y2 % a

def sha3_256_int(data: bytes) -> int:
    """
    sha3_256 hash function wrapper to return integers converted from bytes
    that the hash function returns
    """
    return int.from_bytes(sha3_256(data, usedforsecurity=True).digest(), 'big')

# print(mult_inv(3, 7))   # should return (0, 5) since 3⁻¹ mod 7 = 5
# print(mult_inv(10, 17)) # should return (0, 12) since 10⁻¹ mod 17 = 12
# print(mult_inv(2, 4))   # should return 1 (FAILURE, no inverse)