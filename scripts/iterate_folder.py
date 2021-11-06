# this is an example of iterating over all zst files in a single folder,
# decompressing them and reading the created_utc field to make sure the files
# are intact. It has no output other than the number of lines

import zstandard
import os
import json
import sys
from datetime import datetime
import logging.handlers


log = logging.getLogger("bot")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


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

if __name__ == '__main__':
	input_folder = sys.argv[1]
	output_folder = sys.argv[2]
	# Whatchout of case sensitivity in subreddit names.
	# For example, as below, some starts with upper letters and others with small
	subreddit_list = ['canada', 'liberal', 'conservative','politics']
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

	total_lines = 0
	bad_lines = 0
	total_bytes_processed = 0
	for input_file in input_files:
		file_lines = 0
		file_bytes_processed = 0
		beginflag = True
		created = None
		output_handlers = {}
		comma_flags = {}
		for sub in subreddit_list:
			output_handlers[sub] = open(output_folder+'/'+\
				output_files[file_number]+sub+'.json', 'w')
			output_handlers[sub].write('[')
			comma_flags[sub] = True

		for line, file_bytes_processed in read_lines_zst(input_file[0]):
			try:
				obj = json.loads(line)
				created = datetime.utcfromtimestamp(int(obj['created_utc']))
				sub = obj['subreddit'].lower()
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
		total_lines += file_lines
		total_bytes_processed += input_file[1]
		log.info(f"{created.strftime('%Y-%m-%d %H:%M:%S')} : {total_lines:,} : 100% : {(total_bytes_processed / total_size) * 100:.0f}%")

	log.info(f"Total: {total_lines}")

# An example run:
# python ./iterate_folder.py ../data/2016_submissions/ ../data/processed_new/
