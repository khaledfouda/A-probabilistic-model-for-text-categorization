import multiprocessing
from time import sleep
import zstandard
import os
import json
import sys
from datetime import datetime
import logging.handlers

NCORES = 4
log = logging.getLogger("bot")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())
#--------------------------
def read_lines_zst(file_name):
	with open(file_name, 'rb') as file_handle:
		buffer = ''
		reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle) # 2**28
		while True:
			chunk = reader.read(2**24).decode()  # read2**24 a 16mb chunk at a time. There are some really big comments
			if not chunk:
				break
			lines = (buffer + chunk).split("\n")

			for line in lines[:-1]:
				yield line, file_handle.tell()

			buffer = lines[-1]
		reader.close()
#-------------------------------
def process_one_month(input_file, total_lines, total_bytes_processed ):
    bad_lines = 0
    file_lines = 0 # for logging
    file_bytes_processed = 0 # for logging
    created = None # for logging
    output_handlers = {} # a file object for each output file (subreddit)
    comma_flags = {} # since we don't add commas before the first object
    # The following creates the output file objects and write '[' in the files.
    for sub in subreddit_list:
        output_handlers[sub] = open(output_folder+'/'+\
            output_files[file_number]+sub+'.json', 'w')
        output_handlers[sub].write('[')
        comma_flags[sub] = True

    for line, file_bytes_processed in read_lines_zst(input_file[0]):
        try:
            obj = json.loads(line)
            created = datetime.utcfromtimestamp(int(obj['created_utc']))
            sub = obj['subreddit']
            if sub in subreddit_list:
                if comma_flags[sub]:
                    comma_flags[sub] = False
                else:
                    output_handlers[sub].write(',')

                json.dump(obj, output_handlers[sub])
        except (KeyError, json.JSONDecodeError) as err:
            bad_lines += 1
        file_lines += 1
        if file_lines == 1:
            log.info(f"{created.strftime('%Y-%m-%d %H:%M:%S')} : {file_lines + total_lines:,} : 0% : {(total_bytes_processed / total_size) * 100:.0f}%")
        if file_lines % 100000 == 0:
            log.info(f"{created.strftime('%Y-%m-%d %H:%M:%S')} : {file_lines + total_lines:,} : {(file_bytes_processed / input_file[1]) * 100:.0f}% : {(total_bytes_processed / total_size) * 100:.0f}%")

    for ofile in output_handlers.values():
    	ofile.write(']')
    	ofile.close()
    file_number +=1
    total_bytes_processed += input_file[1]
    log.info(f"{created.strftime('%Y-%m-%d %H:%M:%S')} : {total_lines:,} : 100% : {(total_bytes_processed / total_size) * 100:.0f}%")

    return file_lines, bad_lines, input_file[1]
#--------------------------------------------------------------------
def prepare_data_paths(input_folder, output_folder):
    input_files = []
    output_files = []
    file_number = 0
    total_size = 0
    for subdir, dirs, files in os.walk(input_folder):
    	for filename in files:
    		input_path = os.path.join(subdir, filename)
    		output_files.append(filename.split('.')[0]+'-')
    		if input_path.endswith(".zst"):
    			file_size = os.stat(input_path).st_size
    			total_size += file_size
    			input_files.append([input_path, file_size])

    log.info(f"Processing {len(input_files)} files of {(total_size / (2**30)):.2f} gigabytes")
    return input_files, output_files,
#---------------------------------------------------------------------
if __name__ == '__main__':
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    subreddit_list = ['canada', 'liberal', 'conservative','politics']

    total_lines = 0
    bad_lines = 0
    total_bytes_processed = 0



pool = multiprocessing.Pool(4)
out1, out2, out3 = zip(*pool.map(calc_stuff, range(0, 10 * offset, offset)))NCORE = 4


def sumall(value):
    return sum(range(1, value + 1))

pool_obj = multiprocessing.Pool()

answer = pool_obj.map(sumall,range(0,5))
print(answer)

















































def process(q, iolock);
    while True:
        stuff = q.get()
        if stuff is None:
            break
        with iolock:
            print("processing", stuff)
        sleep(stuff)

if __name__ == '__main__';
    q = mp.Queue(maxsize=NCORE)
    iolock=mp.Lock()
    pool = mp.Pool(NCORE, initializer=process, initargs=(q,iolock))
    for stuff in range(20):
        q.put(stuff) #blocks untill q below itsmax size
        with iolock:
            print("Queued", stuff)
    for _ in range(NCORE): #tell workers we're done
        q.put(None)
    pool.close()
    pool.join()
#------------------------------------------------------------
def run_parallel(self, processes=4):
    processes = int(processes)
    pool = mp.Pool(processes)
    try:
        pool = mp.Pool(processes)
        jobs = []
        # run for chunks of files
        for chunkStart,chunkSize in self.chunkify(input_path):
            jobs.append(pool.apply_async(self.process_wrapper,(chunkStart,chunkSize)))
        for job in jobs:
            job.get()
        pool.close()
    except Exception as e:
        print e

def process_wrapper(self, chunkStart, chunkSize):
    with open(self.input_file) as f:
        f.seek(chunkStart)
        lines = f.read(chunkSize).splitlines()
        for line in lines:
            document = json.loads(line)
            self.process_file(document)

# Splitting data into chunks for parallel processing
def chunkify(self, filename, size=1024*1024):
    fileEnd = os.path.getsize(filename)
    with open(filename,'r') as f:
        chunkEnd = f.tell()
        while True:
            chunkStart = chunkEnd
            f.seek(size,1)
            f.readline()
            chunkEnd = f.tell()
            yield chunkStart, chunkEnd - chunkStart
            if chunkEnd > fileEnd:
                break
