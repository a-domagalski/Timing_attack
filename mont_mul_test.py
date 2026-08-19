import random
import threading

global BASE
BASE = 2**64


# # working parameters of the algorithm
# m = 16381  # modulus
BIT_WIDTH = 64  # BIT_WIDTH of calculation (hexadecimal)
# #BIT_WIDTH = 8  # bytewise calculation
# base = 2**BIT_WIDTH #number system base (hexadecimal)
# length = 4  # in hexadecimal
# #length = 2  # bytes
# R = base**length
# x  = 0x16A0  # 5792 in decimal
# y  = 0x04CD  # 1229 in decimal

def limbsNr(x):
    return (x.bit_length() + 63) // 64

def rand_mod_n(n):
    messageGenerator = sampleMessages(n, 1500)
    messageGenerator.run()
    return messageGenerator.join()

def calc_ro_square(n):
    bits = 2 * limbsNr(n) * 64
    res = 1
    for i in range(0, bits):
        res = res * 2
        while(res > n):
            res = res - n
    return res
        
class sampleMessages (threading.Thread):
    def __init__(self, modulus_n, sampleSize):
        threading.Thread.__init__(self)
        self.modulus_n = modulus_n
        self.sampleSize = sampleSize

    def run(self):
        ro_s = calc_ro_square(self.modulus_n)
        omega = getOmega(self.modulus_n)
        messages = []
        while len(messages) < self.sampleSize:
            m = random.getrandbits(self.modulus_n.bit_length())
            while m > self.modulus_n :
                m = random.getrandbits(self.modulus_n.bit_length())
            messages.append(m)
        self.messages = messages
    def join(self):
        return self.messages
    
# def test(n):
#     omega = getOmega(n)

#     count = 0
#     total = 10000
#     a_list = rand_mod_n(n)
#     b_list = rand_mod_n(n)

#     for i, a in enumerate(a_list):
#         b = b_list[i]
#         low_limb_hits = 0

#         # for i, a in enumerate(a_list):
#         #     b = b_list[i]

#         #     if (a & ((1<<64)-1)) > (1<<63):
#         #         low_limb_hits += 1

#         # print("High low-limb ratio:", low_limb_hits / total)

#         _, extra = MontMul(a, b, n, omega)  # NOT Montgomery domain

#         if extra:
#             count += 1

#     print("extra rate (raw):", count / total)
#     return count / total

def test(n):
    # BIT_WIDTH = 64
    # base = 2 ** BIT_WIDTH

    # number of limbs
    length = (n.bit_length() + BIT_WIDTH - 1) // BIT_WIDTH

    R = pow(BASE, length, n)

    m_neginv = neg_inv(n, BIT_WIDTH, BASE)

    count = 0
    total = 10000

    a_list = rand_mod_n(n)
    b_list = rand_mod_n(n)

    for i, a in enumerate(a_list):
        b = b_list[i]

        # convert to Montgomery domain
        # a_m = convert_to_Montgomery_plain(a, R, n)
        # b_m = convert_to_Montgomery_plain(b, R, n)
        a_m = a
        b_m = b

        _, extra = Montgomery_multiplication(
            a_m, b_m, BASE, n, m_neginv, length
        )

        if extra:
            count += 1

    print("extra rate (Montgomery):", count / total)
    return count / total

def getOmega(n):
    n0 = n & (BASE - 1)   # lowest limb
    inv = pow(n0, -1, BASE)  # modular inverse
    return (-inv) % BASE

# Based on Handbook of Applied Cryptography
# by A. Menezes, P. van Oorschot, and S. Vanstone
# CRC Press, 1996. 
# Chapter 14., 14.36 Algorithm Montgomery multiplication, page 603.




# print('Expected result:',x * y % m)

# preparations with plain maths
def convert_to_Montgomery_plain(x, R, m):
    x = x * R % m
    return x

def neg_inv(m, bit_width, base):
    m0 = m % base
    inv = pow(m0, -1, base)
    return (-inv) % base

# multiplication
# def Montgomery_multiplication(x, y, base, m, m_neginv, length):
#     accu = 0
#     for i in range(length):
#         u = accu % base + (x % base) * (y % base)
#         v = u * m_neginv % base  # modulo base gives the last digit
#         accu = (accu + (x % base) * y + v * m) // base
#         x = x >> BIT_WIDTH  # shiftig x so that modulo base we get the last digit in the next round
#     extra = 0
#     if accu > m:
#         accu -= m
#         extra = 1
#         print("HIT!")
#     return accu, extra

def Montgomery_multiplication(x, y, base, m, m_neginv, length):
    accu = 0

    for i in range(length):
        x_i = x & (base - 1)

        t = accu + x_i * y
        u = ((t & (base - 1)) * m_neginv) & (base - 1)

        accu = (t + u * m) >> BIT_WIDTH

        x >>= BIT_WIDTH

    extra = 0
    if accu >= m:   # IMPORTANT: >= not >
        accu -= m
        extra = 1

    return accu, extra

# inverse conversion with plain maths
def reverse_from_Montgomery_plain(accu, R, m):
    phi_m = m - 1  # for prime numbers
    R_inv = R ** (phi_m - 1) % m
    accu  = accu * R_inv % m
    return accu

# def MontMul(x, y, n, omega):
#     extra = 0
#     w = 64
#     base = 2**64
#     mods = limbsNr(n)
#     r = 0
#     for i in range( 0, mods):
#         yil = y & (base-1)
#         y = y // 2**64
#         u = (omega * (yil * (x & (base-1))+ r&(base-1) )) & (base-1)
#         temp = x * yil
#         temp2 = n * u

#         r = temp + r + temp2
#         r = r >> w
#     if(r > n):
#         extra = 1
#         r = r -n
#     return r, extra


result = 0
while result == 0:
    result = test(7548632255348767914767613759466088515515664018998852642045826181714901478567454660240523219306985162675940698752201200160830984564739844578747156763777193384812854335275773656308097267672904931482654795857112473722069817175333410952097117091758806669004559772745292089883764985617930227528203970054017461922820077023168508681877712541566852284183939114162771323572545304315781507278941379874561500624026555746480330804750697000542901086825223518070149958374861743181818839261483523043493197968445576684656704933659048367707915163844593296745534884623650200090376275973180952444838154952795301565366548535242512835477560223979579481027040806191133868998646052801614214692338131443166151057389926018388878596051505887630043082209391414232880689)