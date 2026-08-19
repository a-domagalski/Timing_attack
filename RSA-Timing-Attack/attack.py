import sys, subprocess, hashlib, shlex, math, binascii, random, copy
import threading
from multiprocessing import Process


global ro
global ro_s
global omega

global sampleSize
sampleSize = 1500

global BASE
BASE = 2**64

global maxKeySize
maxKeySize = 256

global treshold
treshold = 15

global lookAhead
lookAhead = 3

global interactions
interactions = 0

def getLimb(x, i):
    result = (x >> 64*i) & (BASE-1)
    return result

def calc_ro(modulus_n):
    ro = 1
    while ro < modulus_n:
        ro = ro << 64
    return  ro

def interact(  c ) :
    # print("interact(): Did child terminate? (if None - didn't terminate, else - return code):", target.poll())

    global interactions
    interactions += 1
    target_in.write("{0:X}".format(c) + "\n")
    target_in.flush()

    subprocess_rec_cfm = target_out.readline().strip()
    # print("subprocess_rec_cfm: ", subprocess_rec_cfm)

    subprocess_info = target_out.readline().strip()
    # print("Subprocess info: ", subprocess_info)

    time = target_out.readline().strip()
    # print("received time: ", time)
    time = float(time)
    # print("time: ", time)
    message = int(target_out.readline().strip(), 16)
    # print("message: ",message)
    return (time, message)

def limbsNr(x):
    return (x.bit_length() + 63) // 64

def calc_ro_square(n):
    bits = 2 * limbsNr(n) * 64
    res = 1
    for i in range(0, bits):
        res = res * 2
        while(res > n):
            res = res - n
    return res

def getOmega(n):
    n0 = n & (BASE - 1)   # lowest limb
    inv = pow(n0, -1, BASE)  # modular inverse
    return (-inv) % BASE

def MontMul(x, y, n, omega):
    r = 0
    base = 1 << 64
    mask = base - 1
    mods = limbsNr(n)

    for i in range(mods):
        yi = y & mask
        y >>= 64

        u = ((r + yi * x) & mask) * omega & mask

        r = r + yi * x + u * n
        r >>= 64
        # print("DEBUG MontMul: \nr={}; \nn={}".format(r, n))
        rBiggerOrEqual = r >= n
        # print("Is rBiggerOrEqual: ", rBiggerOrEqual)
    if r >= n:
        return r - n, 1
    else:
        return r, 0

def computeSamples(n):
    ro_s = calc_ro_square(n)
    omega = getOmega(n)

    mont_messages = []
    messages = []
    timings = []
    results = []
    for j in range(0, sampleSize):
        m = random.getrandbits(n.bit_length())
        while m > n :
            m = random.getrandbits(n.bit_length())
        messages.append(m)
        mont_messages.append(MontMul(messages[j], ro_s, n, omega)[0])

        res = interact(m)
        timings.append(res[0])
        results.append(res[1])
    return (messages, mont_messages, timings, results)

# def updateMessages(messages, currentKey, n):
#     currentSet = messages[:]
#     for i in range(0, len(messages)):
#         currentSet[i] = MontMul(currentSet[i], currentSet[i], n, omega)[0]

#     for i in range(0, len(messages)):
#         for j in range(1, len(currentKey)):
#             if(currentKey[j] == "1"):
#                 currentSet[i] = MontMul(currentSet[i], messages[i], n, omega)[0]
#             currentSet[i] = MontMul(currentSet[i], currentSet[i], n , omega)[0]
#     return currentSet

def updateMessages(messages, currentKey, n):
    currentSet = []

    for m in messages:
        res = ro % n 
        m_bar = m

        for bit in currentKey:
            res, _ = MontMul(res, res, n, omega)
            if bit == "1":
                res, _ = MontMul(res, m_bar, n, omega)

        currentSet.append(res)

    return currentSet

def checkKey( message, result, d):
    key_1 = int(d + "1", 2)
    key_0 = int(d + "0", 2)

    # choose which prediction was right and output key
    if pow(message, key_1, modulus_n) == result:
        return (1, "1")

    if pow(message, key_0, modulus_n)== result:
        return (1, "0")

    return (0, "0")

def getKeySize(n):
    omega = getOmega(n)
    ro = calc_ro(n)
    print("Getting key SIZE")

    message = MontMul(ro, 1, n, omega)[0]

    res = interact(MontMul(1, 1, n, omega)[0])
    print (res)
    print (MontMul(1, 1, n, omega)[0])
    print ("key size recovery finished")
    # return keySize

class sampleMessages (threading.Thread):
    def __init__(self, modulus_n, sampleSize):
        threading.Thread.__init__(self)
        self.n = modulus_n
        self.sampleSize = sampleSize
    def run(self):
        ro_s = calc_ro_square(self.n)
        omega = getOmega(self.n)
        messages = []
        while len(messages) < self.sampleSize:
            m = random.getrandbits(modulus_n.bit_length())
            while m > modulus_n :
                m = random.getrandbits(modulus_n.bit_length())
            messages.append(m)
        self.messages = messages
    def join(self):
        return self.messages

def checkAnormal(diff1, diff2):
    if (abs(diff1 - diff2) < treshold) | ((diff1 < 0) & (diff2 < 0)):
        return 1
    return 0

class Group(object):
    def __init__(self):
        self.time = 0.0
        self.size = 0

def simulateStep( mont_messages, currentSet, modulus_n, groups, time  ):
    # print("DEBUG: simulateStep(): mont_messages={}, currentSet={}, modulus_n={}, groups={}, time={}".format(mont_messages, currentSet, modulus_n, groups, time))
    encoded_1 = currentSet[:]
    encoded_0 = currentSet[:]
    for i in range(0, len(mont_messages)):
        # assume bit j is 0
        (encoded_0[i], extra) = MontMul(currentSet[i], currentSet[i], modulus_n, omega)
        # if extra reduction
        if extra :
            groups[3].time += time[i]
            groups[3].size += 1
        # if no extra reduction
        else :
            groups[4].time += time[i]
            groups[4].size += 1

        # assume bit = 1
        (temp, extra1) = MontMul(currentSet[i], currentSet[i], modulus_n, omega)  # square
        (encoded_1[i], extra2) = MontMul(temp, mont_messages[i], modulus_n, omega)  # multiply

        # if extra reduction
        if extra1 or extra2 :
            groups[1].time += time[i]
            groups[1].size +=1
        # if no extra reduction
        else :
            groups[2].time += time[i]
            groups[2].size +=1
    count = 0
    #=======================================
    for i in range(len(currentSet)):
        _, extra = MontMul(currentSet[i], currentSet[i], modulus_n, omega)
        if extra:
            count += 1

    print("Extra in square:", count)

    count = 0
    for i in range(len(currentSet)):
        temp, _ = MontMul(currentSet[i], currentSet[i], modulus_n, omega)
        _, extra = MontMul(temp, mont_messages[i], modulus_n, omega)
        if extra:
            count += 1

    print("Extra in multiply:", count)
    #=======================================

    return (groups, encoded_0, encoded_1)

def test(n):
    omega = getOmega(n)

    count = 0
    total = 10000

    for _ in range(total):
        # a = random.randint(1, n-1)
        # b = random.randint(1, n-1)

        a = n - 1
        b = n - 1

        _, extra = MontMul(a, b, n, omega)  # NOT Montgomery domain

        if extra:
            count += 1

    print("extra rate (raw):", count / total)

def avg(group):
    if group.size == 0:
        print("Group size is 0")
        return 0.0
    else:
        print("group size is ", group.size)
    return float(group.time) / group.size

def compute_differences( groups ):

    # compute averaget time for all 4 groups
    # uF1 = float(groups[1].time)/groups[1].size
    # uF2 = float(groups[2].time)/groups[2].size
    # uF3 = float(groups[3].time)/groups[3].size
    # uF4 = float(groups[4].time)/groups[4].size
    print("DEBBUG: compute_differences(): groups[1].time: ", groups[1].time)
    print("DEBBUG: compute_differences(): groups[2].time: ", groups[2].time)
    print("DEBBUG: compute_differences(): groups[3].time: ", groups[3].time)
    print("DEBBUG: compute_differences(): groups[4].time: ", groups[4].time)
    uF1 = avg(groups[1])
    print("DEBBUG: compute_differences(): uF1: ", uF1)
    uF2 = avg(groups[2])
    print("DEBBUG: compute_differences(): uF2: ", uF2)
    uF3 = avg(groups[3])
    print("DEBBUG: compute_differences(): uF3: ", uF3)
    uF4 = avg(groups[4])
    print("DEBBUG: compute_differences(): uF4: ", uF4)


    # compute differences between pari groups
    diff1 = uF1 - uF2
    diff2 = uF3 - uF4
    print("DEBBUG: compute_differences(): diff1: ", diff1)
    print("DEBBUG: compute_differences(): diff2: ", diff2)
    return (diff1, diff2)

def look_Ahead(bitCheck, encoded_0, encoded_1, messages, results, mont_messages, modulus_n, time, key):
    global lookAhead

    key += str(bitCheck)
    (validKey, bit) = checkKey(messages[0], results[0], key)
    if validKey :
        key += str(bit)
        return (validKey, key, delta)

    if bitCheck :
        temp_encoded  = encoded_1[:]
    else:
        temp_encoded  = encoded_0[:]
    delta = 0
    for y in range(0, lookAhead):
        groups  = [ Group() for i in range(5)]

        (groups, encoded_0, encoded_1 ) = simulateStep( mont_messages, temp_encoded, modulus_n, groups, time );

        (diff1, diff2) = compute_differences( groups )

        bit = 0
        if diff1 > diff2:
            bit = 1

        if bit :
            temp_encoded = encoded_1[:]
        else:
            temp_encoded = encoded_0[:]
        key += str(bit)
        # print "ver1 : ", len(key_0), ": ",  bit, "------", abs(int(diff1 - diff2)), "--------", int(diff1), int(diff2), "error:", error
        (validKey, bit) = checkKey(messages[0], results[0], key)
        if validKey :
            key += str(bit)
            return (validKey, key, delta)

        if not checkAnormal(diff1, diff2) :
            delta += abs(int(diff1 - diff2))

    return (validKey, key, delta)

def attack(modulus_n, e):
    global sampleSize

    # set start conditions
    validKey = 0
    keySize  = 64
    foundKey = "1"

    # while the key is within the limits of the maximum key and it is not valid
    while (keySize < maxKeySize) &  (not validKey):
        print("DEBUG: attack(): while loop start")
        # re-initialise everything
        messages        = []
        mont_messages   = []
        currentSet      = []
        results         = []
        time            = []
        stablekeySet    = 0
        error           = 0

        # generate new sample messages of modulus_n bits
        generateMessages = sampleMessages(modulus_n, sampleSize)
        generateMessages.run()

        # add new messages to the current set of messages
        messages += generateMessages.join()

        # reset key to stable key
        currentKey = foundKey

        # compute times taken to decrypt each message
        print("Refer to subprocess (get time and encrypted message for each sample message)")
        for i in range(0, len(messages)):

            # get times for each message ( res[0] is the time taken to decrypt, and res[1] is decrypted message)
            res = interact(messages[i])
            results.append(res[1])
            time.append( res[0] )

            # transform each message into montgomery form
            mont_messages.append(MontMul(messages[i], ro_s, modulus_n, omega)[0])

        print ("Sample Size: ", sampleSize)
        print ("Interactions: ", interactions)
        print ("Start from key bit ", len(currentKey))

        currentSet = updateMessages(mont_messages, currentKey, modulus_n)
        encoded_1 = currentSet[:]
        encoded_0 = currentSet[:]

        # discover the next bits untill the last bit which needs to be guessed
        while ((len( currentKey ) <= keySize) & (error < 7)):
            warning = 0
            groups  = [ Group() for i in range(5)]

            (groups, encoded_0, encoded_1 ) = simulateStep( mont_messages, currentSet, modulus_n, groups, time )

            (diff1, diff2) = compute_differences( groups )

            # condition for anormal behaviour
            if checkAnormal(diff1, diff2) :
                warning = 1
                error += 2
            else :
                if error > 0:
                    error -= 1
                warning = 0

            if warning > 0 :
                print ("warning at bit", len( currentKey ))
                stablekeySet = 1

                # check following rounds for bit 0
                (validKey, possibleKey, delta0) = look_Ahead(0, encoded_0, encoded_1, messages, results, mont_messages, modulus_n, time, currentKey)
                if(validKey):
                    return possibleKey

                # check following rounds for bit 1
                (validKey, possibleKey, delta1) = look_Ahead(1, encoded_0, encoded_1, messages, results, mont_messages, modulus_n, time, currentKey)
                if(validKey):
                    return possibleKey

                # decide which set was better
                if(delta0 < delta1) :
                    bit = 1
                else :
                    bit = 0

            else :
                #if diff for the groups with assumed bit = 1 is bigger than the diff of the groups with assumed bit = 0, then predict 1, else predict 0
                if diff1 > diff2:
                    bit = 1
                else :
                    bit = 0

                if(not stablekeySet):
                    foundKey = currentKey

            # depending on which bit is predicted, keep the results computed with that bit for the next round
            if bit == 1:
                currentSet = encoded_1[:]
            else:
                currentSet = encoded_0[:]

            # add the predicted bit to the key
            currentKey += str(bit)
            (validKey, bit ) = checkKey(messages[0], results[0], currentKey)
            if validKey:
                return currentKey + str(bit)

        else :
            keySize += 16
            sampleSize += 1000
    return currentKey

if ( __name__ == "__main__" ) :

    # if len(sys.argv) < 3 :
    #   raise Exception("not enough argv")

    # inputFile = open(sys.argv[2])
    # modulus_n = int(inputFile.readline(), 16)
    # e = int(inputFile.readline(), 16)
    # inputFile.close()

    if len(sys.argv) < 2 :
      raise Exception("not enough argv")

    print("sys.argv[ 1 ]: ", sys.argv[ 1 ])
    print("Making a sub process representing the attack target.")
  # Produce a sub-process representing the attack target.
    target = subprocess.Popen( args=sys.argv[ 1 ],
                             stdout=subprocess.PIPE,
                             stdin=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             text=True )
    print("Constructing handles to attack target standard input and output.")
  # Construct handles to attack target standard input and output.
    target_out = target.stdout
    target_in  = target.stdin

    print("Reading modulus n and e values from subprocess stdout")
    print("Did child terminate? (if None - didn't terminate, else - return code):", target.poll())
    Info = target_out.readline().strip()
    print("Initial child process info: ", Info)

    modulus_n =  int(target_out.readline().strip(), 16)

    exit()
    e = int(target_out.readline().strip(), 16)
    # print("Modulus n value: {0}, public component e: {1}".format(modulus_n, e))
    key = "1"

    global ro
    ro   = calc_ro(modulus_n)

    global ro_s
    ro_s = calc_ro_square(modulus_n)

    global omega
    omega = getOmega(modulus_n)
    # Execute a function representing the attacker.
    print("starting attack analysis")
    key = attack(modulus_n, e)
    print("attack analysis ended")
    target.close()
    print( "FOUND KEY = ", "{0:X}".format(int(key, 2)))
