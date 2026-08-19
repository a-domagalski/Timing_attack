import custom_rsa
import linecache
import random
import sys
import time
import os
# def read_random_line() -> str:
#     lines_num = 0
#     with open("data/Harry_Potter_and_the_Sorcerer's_ Stone.txt", "r", encoding="utf-8") as f:
#         lines_num = sum(1 for _ in f)
#         f.close()

#     line= ""
#     with open("data/Harry_Potter_and_the_Sorcerer's_ Stone.txt", "r", encoding="utf-8") as f: 
#         for _ in range(random.randint(1, lines_num)):
#             line= f.readline()
#         if line == '\n':
#             print("empyt line reading next one. Line: ", line)
#             line= f.readline()
#     return line.encode("utf-8")

def can_encrypt_chunk(chunk_bytes, n):
    m = int.from_bytes(chunk_bytes, "big")
    return m < n

def chop_line_to_n(line, n: int) -> str:
    chunk_bytes = line
    chunk_bytes_rest = b''
    while True:
        if not can_encrypt_chunk(chunk_bytes, n):
            chunk_bytes_rest += chunk_bytes[-1:]
            chunk_bytes = chunk_bytes[0:-1]
            # print("line after truncating", chunk_bytes)
            # print("rest of the line", str(chunk_bytes_rest))
        else:
            break
    return chunk_bytes, chunk_bytes_rest

def save_key_to_file(key:str):
    with open("subprocess_key.txt", 'w') as file:
        file.write(key)

def get_key_handler():
    customRsa = custom_rsa.CustomRSA()
    n, e, d, p, q = customRsa._generate_keys(nlen=1024)
    keyHandler = custom_rsa.KeyHandler(n=n, e=e, d=d, p=p, q=q)
    return keyHandler        

print("STARTING KEY GENERATION", flush=True)
sys.stdout = open(os.devnull, 'w') #supression of unwanted prints by directing stdout to devnull aka "black hole"
KEY_HANDLER_OBJ = get_key_handler()
save_key_to_file("n: " + str(KEY_HANDLER_OBJ.n) + "e: " + str(KEY_HANDLER_OBJ.e) + "d: " + str(KEY_HANDLER_OBJ.d))

sys.stdout = sys.__stdout__ #stop redirecting stdout to devnull
print(KEY_HANDLER_OBJ.n, flush=True)
print(KEY_HANDLER_OBJ.e, flush=True)

for LINE in sys.stdin:
    print("Receive cfm. LINE: ", LINE, flush=True)
    sys.stdout = open(os.devnull, 'w') #supression of unwanted prints by directing stdout to devnull aka "black hole"
    
    LINE_STRIP = LINE.strip()
    LINE_INT = int(LINE_STRIP, 16)
    LINE_BYTES = LINE_INT.to_bytes((LINE_INT.bit_length() + 7) // 8, "big")

    CHOPPED_LINE, LINE_REST = chop_line_to_n(LINE_BYTES, KEY_HANDLER_OBJ.n)
    print("SUBPROCESS: After input line chopping: CHOPPED_LINE: {}; LINE_REST: {}".format(CHOPPED_LINE, LINE_REST), flush=True)
    ENCRYPTED_MESSAGE = KEY_HANDLER_OBJ.pub.encrypt(CHOPPED_LINE)

    start = time.perf_counter()
    KEY_HANDLER_OBJ.priv.decrypt(ENCRYPTED_MESSAGE)
    end = time.perf_counter()
    
    sys.stdout = sys.__stdout__ #stop redirecting stdout to devnull
    print(str(end - start), flush=True)
    print(ENCRYPTED_MESSAGE, flush=True)