import os
import json

# default exp
# traj_dir = "/Users/anu/Documents/GitHub/routing/main/tool-response-reflection/march14/traj/"
# models = ["gpt-4o","llama-3-3-70b", "mixtral-8x22B"]


# obfuscation
# traj_dir = "/Users/anu/Documents/GitHub/routing/main/tool-response-reflection/obfuscation_experiment_results/traj/"
# models = ["gpt-4o","llama-3-3-70b", "mixtral-8x22B"]

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


datadir = f"{traj_dir}3_updated_with_source"
output_dir = f"{traj_dir}4_parser"
os.makedirs(output_dir, exist_ok=True)


def convert_json_to_jsonl(input_filename, output_filename):
    # Read JSON
    print(input_filename, output_filename)
    with open(input_filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []
    for data_point in data:
        new_data_point = data_point
        if (
            data_point["predicted_output"] is None
            or data_point["predicted_output"] == ""
        ):
            new_data_point["generated_text"] = json.dumps([])
        else:
            new_data_point["generated_text"] = json.dumps(
                data_point["predicted_output"]
            )
        new_data_point["dataset_name"] = new_data_point["metadata"]["dataset"]
        if data_point["metadata"]["ignore"] is True:
            continue
        new_data.append(new_data_point)

    # Open output file in write mode
    with open(output_filename, "w", encoding="utf-8") as out:
        # If the top-level JSON is a list, iterate and write each item on its own line
        if isinstance(new_data, list):
            for item in new_data:
                out.write(json.dumps(item) + "\n")
        # If the top-level JSON is a single object, just write it in one line
        elif isinstance(new_data, dict):
            out.write(json.dumps(new_data) + "\n")
        else:
            print("JSON structure not recognized (not a list or dict).")


for model in models:
    datadir_new = os.path.join(output_dir, model)
    print(datadir_new)
    os.makedirs(datadir_new, exist_ok=True)
    datafiles = os.listdir(os.path.join(datadir, model))
    print(datafiles)
    for item in datafiles:
        if "DS_Store" in item:
            continue

        print(item)

        new_data_file = item + ".jsonl"
        if os.path.isdir(os.path.join(datadir, model, item)):
            continue
        convert_json_to_jsonl(
            os.path.join(datadir, model, item), os.path.join(datadir_new, new_data_file)
        )
