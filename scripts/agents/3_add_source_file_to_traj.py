import json
import os

# default traj - step 2 results
# models = ["gpt-4o","llama-3-3-70b", "mixtral-8x22B"]
# traj_dir = "/Users/anu/Documents/GitHub/routing/main/tool-response-reflection/march14/traj/"

# # obfuscation traj
# models = ["gpt-4o","llama-3-3-70b", "mixtral-8x22B"]
# traj_dir = "/Users/anu/Documents/GitHub/routing/main/tool-response-reflection/obfuscation_experiment_results/traj/"

# # shortlisting traj
models = [
    "gpt-4o_10pct",
    "gpt-4o_25pct",
    "gpt-4o_50pct",
    "gpt-4o_75pct",
    "llama-3-3-70b_10pct",
    "llama-3-3-70b_25pct",
    "llama-3-3-70b_50pct",
    "llama-3-3-70b_75pct",
    "mixtral-8x22B_10pct",
    "mixtral-8x22B_25pct",
    "mixtral-8x22B_50pct",
    "mixtral-8x22B_75pct",
]
traj_dir = "/Users/anu/Documents/GitHub/routing/main/tool-response-reflection/Shortlisting/traj/"

traj_dir_with_source_json = f"{traj_dir}3_updated_with_source/"


print("hi!")
for model in models:
    print("model", model)
    for file in os.listdir(traj_dir + model):
        print("file..", file)
        data = []
        if file == ".DS_Store":
            continue
        with open(f"{traj_dir}{model}/{file}", "r") as f:
            data = json.load(f)

        final_data = []
        for item in data:
            print(item["metadata"]["dataset"])
            item["source_file_path"] = (
                item["metadata"]["dataset"] + "_nestful_format_bird.json"
            )
            final_data.append(item)
        os.makedirs(f"{traj_dir_with_source_json}{model}", exist_ok=True)
        with open(f"{traj_dir_with_source_json}{model}/{file}", "w") as f:
            json.dump(final_data, f)

        print(f"{traj_dir_with_source_json}{model}/{file} created")
