from collections import Counter
from glob import glob
import json
import os
from time import time


def classify_images(directory):
    counts = {}
    numfiles = 0
    start = time()
    num_json = len(glob(os.path.join(directory, "**/*.json")))
    tot_len = len(str(num_json))

    # Iterate over all files in the directory
    for root, _, files in os.walk(directory):
        if files:
            dir = root.split(os.sep)[-1]
            counts[dir] = Counter()
        for filename in files:
            if filename.endswith(".json"):
                numfiles += 1
                if time() - start > 0.1 or num_json == numfiles:
                    prop_done = numfiles/num_json
                    width = os.get_terminal_size().columns - 1
                    starter_text = f"Progress: {str(numfiles).rjust(tot_len)}/{num_json}"
                    width_left = width - len(starter_text) - 3
                    if width_left > 5:
                        loaded = int(width_left * prop_done)
                        loading_bar = f" |{str("="*loaded).ljust(width_left)}|"
                        print("\r" + starter_text + loading_bar, end="")
                    else:
                        print(starter_text, end="\r")
                    start = time()
                file_path = os.path.join(root, filename)

                # Read JSON file
                with open(file_path, "r") as f:
                    data = json.load(f)

                # Extract time of day
                time_of_day = data.get("attributes", {}).get("timeofday", "").lower()
                counts[dir][time_of_day] += 1
                
    # Print counts:
    print()
    for key, val in counts.items():
        print(f"{key}:")
        for countkey in sorted(val.keys()):
            countval = val[countkey]
            print(f"  {str(countkey).ljust(10)}:{str(countval).rjust(6)} {str(round(100*countval/val.total(), 0)).rjust(5)}%")
    
    return counts


# Example usage
directory_path = os.path.expanduser("~/Downloads/bdd100k_labels/")  # Change this to your actual directory

counts = classify_images(directory_path)
with open("day_night_counts.json", "w") as f:
    json.dump(counts, f)
