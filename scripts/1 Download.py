

if __name__ == '__main__':
    import wget

    # Downloading reddit submissions for 2019 and 2020
    # One file per month
    # can be modified to include more years
    url_leading = "https://files.pushshift.io/reddit/submissions/"
    output_folder = '../data/submissions_zst/'
    for year in [2018, 2019]:
        for month in range(1, 13):
            print(f"Downloading reddit submissions for {month:02d}/{year}")
            url = f"{url_leading}/RS_{year:04d}-{month:02d}.zst"
            outfile = f"{output_folder}/RS_{year:04d}-{month:02d}.zst"
            print(f"Saving to {outfile}")
            wget.download(url, outfile)
            print(f"File is saved")
    print("All files are saved.")
