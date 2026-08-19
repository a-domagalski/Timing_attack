from provable_prime import get_seed, generate_provable_prime_pair
from sympy import primerange
from math import gcd
from tools.gen_use_funs import mult_inv
from Crypto.Cipher import PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Hash import SHA256
from Crypto.Math.Numbers import Integer
import time
import math

# Returns:
    # status - The status returned from the generation routine where status is either SUCCESS or FAILURE. 
    #          If FAILURE is returned, then zeros are returned as the other output values.
    # prime - The requested prime.
    # prime_seed -  A seed determined during generation.
    # prime_gen_counter - (Optional) A counter determined during the generation of the prime.

# If you encode text as UTF-32BE (4 bytes/character, no BOM) 
# and you use RSA-OAEP with SHA-256 as the mask/hash for OAEP,
# the maximum number of UTF-32 characters you can encrypt directly is:
# 1024-bit RSA: 62 bytes payload → 15 UTF-32 characters
# 2048-bit RSA: 190 bytes payload → 47 UTF-32 characters
# 3072-bit RSA: 318 bytes payload → 79 UTF-32 characters
# 4096-bit RSA: 446 bytes payload → 111 UTF-32 characters
MAX_ENCRYPT_TXT_SIZE_MAP = {
    1024: 15,
    2048: 47,
    3072: 79,
    4096: 111,
}

global mont_mul_base
mont_mul_base = 2**64

def limbsNr(x):
    return (x.bit_length() + 63) // 64

def getLimb(x, i):
    return (x >> (64 * i)) & (mont_mul_base - 1)

# def getOmega(n):
#     w1 = getLimb(n, 0)
#     w = 1
#     for _ in range(64):
#         w = (w * w * w1) % mont_mul_base
#     return w

def getOmega(n):
    n0 = n & (mont_mul_base - 1)   # lowest limb
    inv = pow(n0, -1, mont_mul_base)  # modular inverse
    return (-inv) % mont_mul_base

def MontMul(x, y, n, omega):
    r = 0
    base = 1 << 64
    mask = base - 1
    mods = limbsNr(n)

    for i in range(mods):
        yi = y & mask
        y >>= 64

        u = ((r + yi * x) & mask) * omega & mask

        r = (r + yi * x + u * n) >> 64

    if r >= n:
        return r - n, 1
    else:
        return r, 0
    
def sqam_montgomery(base, exponent, n):
    w = 64
    k = limbsNr(n)
    R = 1 << (w * k)

    omega = getOmega(n)

    base_bar = (base * R) % n
    res_bar = R % n

    for bit in bin(exponent)[2:]:
        res_bar, _ = MontMul(res_bar, res_bar, n, omega)
        if bit == "1":
            res_bar, _ = MontMul(res_bar, base_bar, n, omega)

    result, _ = MontMul(res_bar, 1, n, omega)
    return result

class CustomRsaKey(RSA.RsaKey):
    '''
    Docstring for CustomRsaKey
    
    Enables use of _decrypt() method from RSA.RsaKey that features
    countermeasurements for timing attacks as well as custom decrypt()
    method that lacks any countermeasurements.
    '''
    def __init__(self, **kwargs):
        """
        Docstring for __init__
        
        To construct objects of type RSA.RsaKey RSA.construct() is used
        so super().__init__() isn't called here (RSA.construct() makes 
        additional computation of params needed in RSA.RsaKey).
        :param self: Description
        :param kwargs: Description
        """
        # self.is_pub = False
        # self.is_priv = False
        # super().__init__(**kwargs)
        # print(self._n, self._e, self._d)

        self.is_pub = getattr(self, 'is_pub', None)
        self.is_priv = getattr(self, 'is_priv', None)
        print("__init__ print ", self.is_pub)
        if hasattr(self, '_e') and not hasattr(self, '_d'):
            print("This is pub key")
            self.is_pub = True
        elif hasattr(self, '_n') and hasattr(self, '_d') and not hasattr(self, '_e'):
            print("This is priv key")
            self.is_priv = True
        # if self._n and self._e and not self._d:
        #     self.is_pub = True
        # if self._n and self._d and not self._e:
        #     self.is_priv = True

    # def __set_initial_prams(self):
    #     """
    #     Docstring for __set_initial_prams
    #     This method is created for possibility to use RSA.construct() method.
    #     RSA.Construct() creates object of type RSA.RsaKey running it's constructor
    #     and setting RSA key parameters. To copy those parameters, to newly created
    #     object of type CustomRsaKey, self.from_parent method is used.
    #     The effect of above is __init__ of CustomRsaKey won't be ran, and it's custom
    #     parametrs wouldn't be set, thus this method is used (cannot run __init__ of 
    #     CustomRsaKey after using RSA.construct() because it runs super().__init__()
    #     ).
    #     :param self: Description
    #     """

    @classmethod
    def from_parent(cls, rsaKey:RSA.RsaKey):
        """
        Docstring for from_parent
        
        :param cls: Not sure, but this is object that represents type of *this* class
        :param rsaKey: Object of parent class to copy parameter values from.
        :type rsaKey: RSA.RsaKey
        """
        customKey = cls.__new__(cls)
        print("from_parent(): Type of __new__ is: ", type(customKey))

        customKey.__dict__ = rsaKey.__dict__.copy()
        print("from_parent(): Type of __new__ is: ", type(customKey))

        return customKey
    
    @property
    def key_size_bytes(self):
        return (self.n.bit_length() + 7) // 8


    # TODO dlugosc message musi byc sprawdzana w bytes i tak samo dla modulus n
    def encrypt(self, plaintext:bytes=b'', K=None):
        if self.is_pub:
            return self._encrypt(int.from_bytes(plaintext, 'big'))
        elif self.is_priv:
            e_place_holder = self._e
            self._e = self._d
            encrypted_plaintext = self._encrypt(int.from_bytes(plaintext, 'big'))
            self._e = e_place_holder
            return encrypted_plaintext

    def decrypt(self, message: int):
        """
        Docstring for decrypt
        This method in opposite to super()._decrypt() lacks any 
        countermeasurements against timing attacks.
        
        :param self: Description
        :param message: Description
        :type message: bytes
        """
        # mess_int = int.from_bytes(message, 'big')
        return int(sqam_montgomery(message, self._d._value, self._n._value)).to_bytes(self.key_size_bytes, "big").lstrip(b"\x00")
    
    # def test():
    #     print("test of inheritance")
    
    def sign(self, message, priv_key):
        if self.is_priv:
            mess_digest = SHA256.new().digest(message)
            start = time.perf_counter()
            encrypted_digest = self.encrypt(message=mess_digest, key_bits=2048, hash_len=32, bytes_per_char=1)
            end = time.perf_counter()
            time_elapsed = end - start
        elif self.is_pub:
            raise ValueError("Trying to sign with public key")

        return encrypted_digest, time_elapsed

class KeyHandler:
    def __init__(self, **kwargs):
        self.pub = None
        self.priv = None

        self.crypto_pub = None
        self.crypto_priv = None
        
        for key, value in kwargs.items():
            setattr(self, key, value)
        if  not hasattr(self, 'n') or\
            not hasattr(self, 'e') or \
            not hasattr(self, 'd'):
            raise AttributeError("'n' or/and 'e' or/and 'd' attribute is/are lacking")
            
        self.__construct_custom_key()
        self.__construct_crypto_key()

    def __construct_crypto_key(self):
        pub = RSA.construct((self.n, self.e))
        priv = RSA.construct((self.n, self.e, self.d))

        self.crypto_pub = pub.publickey().export_key(format='PEM')
        self.crypto_priv = priv.export_key(format='PEM')

    def __construct_custom_key(self):
        pub = RSA.construct((self.n, self.e))
        priv = RSA.construct((self.n, self.e, self.d))
        # self.pub = CustomRsaKey(n=self.n, e=self.e)
        # self.priv = CustomRsaKey(n=self.n, e=self.e, d=self.d)
        
        self.pub = CustomRsaKey.from_parent(pub)
        self.pub.__init__()
        self.priv = CustomRsaKey.from_parent(priv)
        self.priv.__init__()
    
class CustomRSA:
    def __init__(self):
        self._generate_keys()
        self.__export_keys()

    def __find_e(self, totient):
        odd_num = 65537
        if odd_num < totient and gcd(odd_num, totient) == 1:
            return odd_num
        else:
            for num in reversed(range(3, odd_num + 1, 2)):
                if gcd(num, totient) == 1:
                    # print(num)
                    return num
        return False

    def _generate_keys(self, nlen:int=2048):
        status1, seed = get_seed(nlen)
        # print("rsa seed: ", type(seed))
        status2, p, q = generate_provable_prime_pair(nlen, 65537, seed)
        if not status1 or not status2:
            return False

        n = p * q # modulus
        totient = (p - 1) * (q - 1)
        e = self.__find_e(totient=totient)
        status, d = mult_inv(e, totient)

        self.n = n
        self.e = e
        self.d = d

        return n, e, d, p, q
    
    def __export_keys(self):
        """
        keys_tuple - (n, e, d)
        """
        self.pub = RSA.construct((self.n, self.e))
        self.priv = RSA.construct((self.n, self.e, self.d))

        self.pub_pem = self.pub.publickey().export_key(format='PEM')
        self.priv_pem = self.priv.export_key(format='PEM')

    def __get_message_max_len(self, key_bits_len:int=2048, hash_len:int=32, bytes_per_char:int=4):
        k = key_bits_len // 8
        max_payload = k - 2 * hash_len - 2
        max_chars = max_payload // bytes_per_char  # UTF-32 uses 4 bytes/char
        return max_chars

    def __chop_long_message(self, message, max_chars, key_bits_len:int=2048, hash_len:int=32):
        # Compute OAEP max payload for this key size + hash
        # (compute what is the maximum num of chars that we are able to encrypt
        # propperly)

        # Split the message into chunks that fit the RSA-OAEP limit
        chunks = [message[i:i + max_chars] for i in range(0, len(message), max_chars)]
        return chunks
    
    @property
    def key_size_bytes(self):
        return (self.n.bit_length() + 7) // 8
    
    def _encrypt_pkcs1_oaep(self, message: str, key_bits_len:int=2048, hash_len:int=32, bytes_per_char:int=4):
        '''
        Docstring for encrypt

        Encrypts ENCODED message using RSA with OAEP padding 
        and SHA256. Version with timing attack countermeasures
        hash_len: 

        :param self: Description
        :param message: Description
        :type message: str
        :param key_bits_len: Description
        :type key_bits_len: int
        :param hash_len: length of hash function used for PKCS1_OAEP (32 bytes and 256 bits).
                         for computation purposes it's 32 instead of 256
        :type hash_len: int
        :param bytes_per_char: Description
        :type bytes_per_char: Number of bytes needed for one char after encoding the message
        '''
        if self.pub_pem == None or self.priv_pem == None:
            return False
        
        pubkey = RSA.import_key(self.pub_pem)
        cipher = PKCS1_OAEP.new(pubkey, hashAlgo=SHA256)

        if len(message) > self.__get_message_max_len(key_bits_len=key_bits_len, hash_len=hash_len, bytes_per_char=bytes_per_char):
            chunks = self.__chop_long_message(message=message, key_bits_len=key_bits_len, hash_len=hash_len)
            ciphertexts = []
            for chunk in chunks:
                data = chunk.encode('utf-32-be')
                ciphertexts.append(cipher.encrypt(data))
            return ciphertexts
        return cipher.encrypt(message=message)
    
    def _decrypt_pkcs1_oaep(self, ciphertexts, rsa_privkey):
        """
        Decrypts texts inside of ciphertexts list encrypted using RSA with
        OAEP padding and hash SHA256 and encoded with UTF32 
        """
        if self.pub_pem == None or self.priv_pem == None:
            return False
        
        privkey = RSA.import_key(rsa_privkey)
        decipher = PKCS1_OAEP.new(privkey, hashAlgo=SHA256)
        plaintext = ""
        for c in ciphertexts:
            chunk_bytes = decipher.decrypt(c)
            plaintext += chunk_bytes.decode('utf-32-be')
        return plaintext
    
    # TODO szyfrowanie powinno być zrobione z użyciem klucza prywatnego, nie publicznego!
# def find_e(totient):
#     odd_num = 65537
#     if odd_num < totient and gcd(odd_num, totient) == 1:
#         return odd_num
#     else:
#         for num in reversed(range(3, odd_num + 1, 2)):
#             if gcd(num, totient) == 1:
#                 print(num)
#                 return num
#     return False

# def generate_keys():
#     nlen = 2048
#     status1, seed = get_seed(nlen)
#     # print("rsa seed: ", type(seed))
#     status2, p, q = generate_provable_prime_pair(nlen, 65537, seed)
#     if not status1 or not status2:
#         return False

#     n = p * q # modulus
#     totient = (p - 1) * (q - 1)
#     e = find_e(totient=totient)
#     status, d = mult_inv(e, totient)

#     return n, e, d

# # If you encode text as UTF-32BE (4 bytes/character, no BOM) 
# # and you use RSA-OAEP with SHA-256 as the mask/hash for OAEP,
# # the maximum number of UTF-32 characters you can encrypt directly is:
# # 1024-bit RSA: 62 bytes payload → 15 UTF-32 characters
# # 2048-bit RSA: 190 bytes payload → 47 UTF-32 characters
# # 3072-bit RSA: 318 bytes payload → 79 UTF-32 characters
# # 4096-bit RSA: 446 bytes payload → 111 UTF-32 characters
# MAX_ENCRYPT_TXT_SIZE_MAP = {
#     1024: 15,
#     2048: 47,
#     3072: 79,
#     4096: 111,
# }

# def encrypt(text, rsa_pubkey, key_bits=2048, hash_len=32):
#     """
#     Encrypts text encoded with UTF32 using RSA with OAEP padding 
#     and hash SHA256
#     """
#     # Compute OAEP max payload for this key size + hash
#     k = key_bits // 8
#     max_payload = k - 2 * hash_len - 2
#     max_chars = max_payload // 4  # UTF-32 uses 4 bytes/char

#     pubkey = RSA.import_key(rsa_pubkey)
#     cipher = PKCS1_OAEP.new(pubkey, hashAlgo=SHA256)

#     # Split the text into chunks that fit the RSA-OAEP limit
#     chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
#     ciphertexts = []

#     for chunk in chunks:
#         data = chunk.encode('utf-32-be')
#         ciphertexts.append(cipher.encrypt(data))
#     return ciphertexts

# def decrypt(ciphertexts, rsa_privkey):
#     """
#     Decrypts texts inside of ciphertexts list encrypted using RSA with
#     OAEP padding and hash SHA256 and encoded with UTF32 
#     """
#     privkey = RSA.import_key(rsa_privkey)
#     decipher = PKCS1_OAEP.new(privkey, hashAlgo=SHA256)
#     plaintext = ""
#     for c in ciphertexts:
#         chunk_bytes = decipher.decrypt(c)
#         plaintext += chunk_bytes.decode('utf-32-be')
#     return plaintext

    


# # test_text = "siala baba mak nie wiedziala jak a dziad wiedzial nie powiedzial i to bylo tak"
# # def decrypt():
# # print(generate_keys())
# N, E, D = generate_keys()
# # pub = (N, E)
# # priv = (N, D)
# pubkey = RSA.construct((N, E))
# # Construct private key (prefer providing p,q for CRT)
# privkey = RSA.construct((N, E, D))

# # Export to PEM (unencrypted)
# pem_priv = privkey.export_key(format='PEM')
# pem_pub = pubkey.publickey().export_key(format='PEM')
# text = encrypt(test_text, pem_pub)
# print(text)
# text = decrypt(text, rsa_privkey=pem_priv)
# print(text)
#================================================================================================
# test_text = "siala baba mak nie wiedziala jak a dziad wiedzial nie powiedzial i to bylo tak"
# test_text_enc = test_text.encode("utf-8")
# customRsa = CustomRSA() # key generation at object creation
# encrypted_text = customRsa.encrypt(message=test_text_enc)
# # print(encrypted_text)

# decrypted_text = customRsa.decrypt(encrypted_text)

# print(decrypted_text.decode("utf-8"))

# print()