from custom_rsa import CustomRSA, MAX_ENCRYPT_TXT_SIZE_MAP
import time
import json, base64
import sys, os
from statistics import mean

# def display_progress_bar(iter_total, current_iter, bar_len=40, do_flush=True):
#     percent = current_iter / iter_total
#     filled = int(bar_len * percent)
#     bar = "█" * filled + "-" * (bar_len - filled)
#     sys.stdout.write(f"\rProgress: |{bar}| {percent*100:6.2f}%")
#     if do_flush:
#         sys.stdout.flush()

def construct_progress_bar(iter_total, current_iter, bar_len=40, do_flush=True):
    percent = current_iter / iter_total
    filled = int(bar_len * percent)
    bar = "█" * filled + "-" * (bar_len - filled)
    return f"\rProgress: |{bar}| {percent*100:6.2f}%"


def display_in_place(data_to_display, len_prev_lines):
    if len_prev_lines:
        sys.stdout.write(f"\033[{len_prev_lines}F")  # move up N lines
        for i in range(len_prev_lines):
            sys.stdout.write("\033[K")  # clear line
            sys.stdout.write("\n")  # Moves cursor to start of next line
        sys.stdout.write(f"\033[{len_prev_lines}F")  # move up N lines

    for data in data_to_display:
        sys.stdout.write(data)
    sys.stdout.flush()



def read_generate_write(filepath_read, file_encoding, filepath_write, chunk_size):
    file_size = os.path.getsize(filepath_read)
    with open(filepath_write, "w") as f_save:
        with open(filepath_read, "r", encoding=file_encoding) as f_read:
            read_so_far = 0
            exec_times = []
            first_iter = True
            time_total = 0
            while True:
                progress_bar = construct_progress_bar(iter_total=file_size, current_iter=read_so_far, bar_len=100, do_flush=False)
                data_to_display = []
                chunk = f_read.read(chunk_size)
                read_so_far += len(chunk.encode("utf-8"))
                if not chunk:
                    break  # End of file reached
                n, e, d, text, encrypted_text, elapsed_time = generate_data(chunk)
                exec_times.append(elapsed_time)
                time_total += elapsed_time
                record = {"n": n, "e": e, "d": d, "text": text, "encrypted_text": base64.b64encode(encrypted_text).decode("ascii"), "elapsed_time_s": elapsed_time, "key_size": KEY_SIZE}
                
                data_to_display.append(progress_bar)
                data_to_display.append("\n")
                data_to_display.append(f"Elapsed time: {elapsed_time}")
                data_to_display.append("\n")
                data_to_display.append("Time mean=" + str(mean(exec_times)))
                data_to_display.append("\n")
                data_to_display.append("Time total=" + str(time_total))
                len_prev_lines = len(data_to_display) - 4 # -4 because there are newlines so there are 4 lines displayed in total (there are 3 "\n" but with -4 it works, dunno why)
                if first_iter:
                    len_prev_lines = 0
                display_in_place(data_to_display=data_to_display, len_prev_lines=len_prev_lines)
                
                json.dump(record, f_save)
                f_save.write("\n")
                first_iter = False

def generate_data(text):
    start_time = time.time()
    customRsa = CustomRSA() # key generation at object creation
    encrypted_text = customRsa.encrypt(text=text, key_bits=2048)
    print(len(encrypted_text))
    if len(encrypted_text) == 1:
        encrypted_text = encrypted_text[0]
    end_time = time.time()
    elapsed_time = end_time - start_time
    return customRsa.n, customRsa.e, customRsa.d, text, encrypted_text, elapsed_time

KEY_SIZE = 4096
CHUNK_SIZE = MAX_ENCRYPT_TXT_SIZE_MAP[KEY_SIZE]
FILEPATH_READ = "D:/Projects/NNRSAAttack/data/Harry_Potter_and_the_Sorcerer's_ Stone.txt"
FILE_ENCODING = "utf-8"
FILEPATH_WRITE = "D:/Projects/NNRSAAttack/data/rsa_data.json"
read_generate_write(filepath_read=FILEPATH_READ, file_encoding=FILE_ENCODING, filepath_write=FILEPATH_WRITE, chunk_size=CHUNK_SIZE)