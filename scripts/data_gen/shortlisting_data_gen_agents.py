import json
import os
from typing import List


# run experiments for 4 different %'s

expt_name = "pct75"
# 1. Identify all JSON files in the current directory.
# json_files = [f for f in os.listdir('.') if f.endswith('.json')]
datadir = "/Users/anu/Documents/GitHub/routing/main/tool-response-reflection/data_rest"

json_files = os.listdir(datadir)
all_rows: List = []

data_points_math = {}


for filename in json_files:
    with open(os.path.join(datadir, filename), "r", encoding="utf-8") as f:
        data = json.load(f)
    datapoints: List = []
    for datapoint in data:
        output = datapoint["output"]
        tools = datapoint["tools"]

        print("Length of tools", len(tools))
        # Pick 20 distinct dictionaries at random
        # (Make sure len(data) >= 20!)
        pct10 = int(len(tools) * 10 / 100)
        pct25 = int(len(tools) * 25 / 100)
        pct50 = int(len(tools) * 50 / 100)
        pct75 = int(len(tools) * 75 / 100)
        data_points_math[filename] = {
            "pct10": pct10,
            "pct25": pct25,
            "pct50": pct50,
            "pct75": pct75,
            "total_tools": len(tools),
        }
        break

print(json.dumps(data_points_math, indent=4))


datadir_new = f"/Users/anu/Documents/GitHub/routing/main/tool-response-reflection/test_data_rest_{expt_name}"
os.makedirs(datadir_new, exist_ok=True)
for filename in json_files:
    with open(os.path.join(datadir, filename), "r", encoding="utf-8") as f:
        data = json.load(f)
    datapoints = []
    for datapoint in data:
        output = datapoint["output"]
        tools = datapoint["tools"]

        # print("Length of tools", len(tools))
        # Pick 20 distinct dictionaries at random
        # (Make sure len(data) >= 20!)

        import random

        # The data point we want to exclude
        exclude_data_point = output

        # Filter out the excluded data point
        filtered_data = [d for d in tools if d != exclude_data_point]

        random_sample = random.sample(
            filtered_data, data_points_math[filename][expt_name] - 1
        )
        print("len of random sample", len(random_sample), filename)
        save_datapoint = {}
        for item in tools:
            if item["name"] == output[0]["name"]:
                save_datapoint = item
                break
        if len(save_datapoint) != 0:
            random_sample.append(save_datapoint)
        # print(len(random_sample))
        print(
            "len of random sample - appended",
            len(random_sample),
            datapoint["sample_id"],
            save_datapoint,
        )
        random.shuffle(random_sample)

        new_datapoint = datapoint
        new_datapoint["tools"] = random_sample
        print("new_tools", len(random_sample))
        datapoints.append(new_datapoint)

    with open(os.path.join(datadir_new, filename), "w") as f:
        data_updated = {"data": datapoints}
        json.dump(datapoints, f)
