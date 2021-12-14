

if __name__ == '__main__':
    import wget
    import sys
    year_list = sys.argv[:]
    if len(year_list) == 0:  # Default value
        year_list = [2019, 2020]
    # One file per month
    url_leading = "https://files.pushshift.io/reddit/submissions/"
    output_folder = '../data/submissions_zst/'
    for year in year_list:
        for month in range(1, 13):
            print(f"Downloading reddit submissions for {month:02d}/{year}")
            url = f"{url_leading}/RS_{year:04d}-{month:02d}.zst"
            outfile = f"{output_folder}/RS_{year:04d}-{month:02d}.zst"
            print(f"Saving to {outfile}")
            wget.download(url, outfile)
            print(f"File is saved")
    print("All files are saved.")
# Example run: # python 1_download.py 2018 2019 2020
