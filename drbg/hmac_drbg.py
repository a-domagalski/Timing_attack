from drbg.entropy import get_entropy_input
from hashlib import sha3_256
# from tools.gen_use_funs import bytes_to_bits
import hmac as std_hmac
from tools.states import FAILURE, SUCCESS
from tools.gen_use_funs import bits_to_bytes, bytes_to_bits

class Hmac_drbg:
    OUTLEN = 256
    BLOCK_SIZE = 136 #for sha3 256 block size is 136 bytes https://en.wikipedia.org/wiki/Secure_Hash_Algorithms 

    def __init__(self, requested_instantiation_security_strength=None, personalization_string=""):
        self.key = None
        self.v = None
        self.reseed_counter = 0
        self.security_strength = None
        self.requested_instantiation_security_strength = requested_instantiation_security_strength
        self.__hmac_drbg_instantiate_function(requested_instantiation_security_strength, personalization_string)

    # Input: integer (requested_instantiation_security_strength), bitstring personalization_string. 
    # Output: string status, integer state_handle.
    def __hmac_drbg_instantiate_function(self, requested_instantiation_security_strength, personalization_string):
        # Check the validity of the input parameters.
        if requested_instantiation_security_strength > 256:
            return "Invalid requested_instantiation_security_strength", -1
        if len(personalization_string) > 160:
            return "Personalization_string too long", -1
        # Set the security_strength to one of the valid security strengths.
        if requested_instantiation_security_strength <= 112:
            self.security_strength = 112
        elif requested_instantiation_security_strength <= 128:
            self.security_strength = 128
        elif requested_instantiation_security_strength <= 192:
            self.security_strength = 192
        else:
            self.security_strength = 256
        # Get the entropy_input and the nonce.
        min_entropy = 1.5 * self.security_strength
        status, entropy_input = get_entropy_input(min_entropy, 1000)
        if status != SUCCESS:
            # print("failure")
            return status, -1
        # Invoke the instantiate algorithm. Note that the entropy_input contains the nonce.
        self.__hmac_drbg_instantiate_algorithm(entropy_input, personalization_string)
        # (status, state_handle) = Find_state_space() 
        # if status != "Success":
        #     return status, -1

    def __hmac_drbg_instantiate_algorithm(self, entropy_input: str, personalization_string: str):
        seed_material = entropy_input + personalization_string
        outlen_bytes = int(self.OUTLEN/8)
        key = bytes([0x00] * outlen_bytes)
        v = bytes([0x01] * outlen_bytes)
        self.key, self.v = self.__hmac_drbg_update(seed_material, key, v)
        self.reseed_counter = 1
        # x - 0,4,y - 0,4,z - 0,1 ,k - 0,1

    def __hmac_drbg_update(self, porvided_data, key, v):
        key_loc = self.__reference_hmac_sha3_256(key, v + b'\x00' + bits_to_bytes(porvided_data))
        v_loc = self.__reference_hmac_sha3_256(key_loc, v)
        if porvided_data == None:
            return key_loc, v_loc
        key_loc = self.__reference_hmac_sha3_256(key_loc, v_loc + b'\x01' + bits_to_bytes(porvided_data))
        v_loc = self.__reference_hmac_sha3_256(key_loc, v_loc)
        # print("key_loc, v_loc: ", key_loc, v_loc)
        return key_loc, v_loc

    # returns a 32-byte bytes object
    def __reference_hmac_sha3_256(self, key, msg):
        return std_hmac.new(key, msg, sha3_256).digest()
    
    def hmac_drbg_generate_function(self, requested_no_of_bits):
        """
        Calls __hmac_drbg_generate_algorithm() to generate
        pseudorandom_bits, so returns pseudorandom_bits as bitstring, not bytes.
        """
        requested_security_strength = self.requested_instantiation_security_strength
        if requested_no_of_bits > 7500:
            return "Too many bits requested", None
        if requested_security_strength > self.security_strength:
            return "Invalid requested_security_strength", None
        status, pseudorandom_bits, self.v, self.key, self.reseed_counter = self.__hmac_drbg_generate_algorithm(self.v, self.key, self.reseed_counter, requested_no_of_bits)
        if status == "Reseed required": 
            return "DRBG can no longer be used. A new instantiation is required", None
        return SUCCESS, pseudorandom_bits #returns bits
    
    # Input: bitstring (V, Key), integer (reseed_counter, requested_number_of_bits).
    def __hmac_drbg_generate_algorithm(self, v, key, reseed_counter: int, requested_no_of_bits:int):
        """
        Generate and return pseudorandom bits\n
        Returns pseudorandom_bits as bitstring, not bytes
        """
        if reseed_counter >= 10000:
            return "Reseed required", None, v, key, reseed_counter
        v_loc = None
        key_loc = None
        temp = b""
        temp_bits = b""
        while len (temp_bits) < requested_no_of_bits:
            v_loc = self.__reference_hmac_sha3_256(key, v)
            temp = temp + v_loc
            temp_bits = bytes_to_bits(temp)
        # temp = gen_use_funs.bytes_to_bits(temp)
        pseudorandom_bits = temp_bits[:requested_no_of_bits]
        key_loc, v_loc = self.__hmac_drbg_update(None, key, v_loc)
        reseed_counter = reseed_counter + 1.
        return SUCCESS, pseudorandom_bits, v_loc, key_loc, reseed_counter

temp = Hmac_drbg(80)
temp._Hmac_drbg__hmac_drbg_generate_algorithm(bits_to_bytes("0101010101"),bits_to_bytes("0101010111"),0, 1024 )