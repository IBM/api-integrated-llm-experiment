import os
import json
from typing import Dict, Any


# initial agent (step 1 results)

data_points_ignored_file = "/Users/anu/Documents/GitHub/routing/main/api_integrated_llm_experiment_jk_march3/output/datapoints_to_be_ignored.json"


########## default_agent_experiments ##############
# models = ["gpt-4o","llama-3-3-70b","mixtral-8x22B"]
#
# expt = "default_agent_experiments_igf"
# # step 1 raw results from the agent
# agent_results_dir = "/Users/anu/Documents/GitHub/routing/main/tool-response-reflection/march14/"
# # JSON format (step 3 results with source file in it)
# traj_dir = "/Users/anu/Documents/GitHub/routing/main/tool-response-reflection/march14/traj/3_updated_with_source/"
# results_dir = f"/Users/anu/Documents/GitHub/routing/main/tool-response-reflection/default_agent_experiments_igf"

# ########## obfuscation_experiments ##############
# models = ["gpt-4o","llama-3-3-70b","mixtral-8x22B"]

# expt = "obfuscation_experiment"
# # step 1
# agent_results_dir = "/Users/anu/Documents/GitHub/routing/main/tool-response-reflection/obfuscation_experiment_results/"
# # JSON format (step 2 results)
# traj_dir = "/Users/anu/Documents/GitHub/routing/main/tool-response-reflection/obfuscation_experiment_results/traj/"
# results_dir = f"/Users/anu/Documents/GitHub/routing/main/tool-response-reflection/{expt}_results_igf"


# ########## shortlisting_experiments ##############


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
expt = "shortlisting"
# step 1
agent_results_dir = (
    "/Users/anu/Documents/GitHub/routing/main/tool-response-reflection/Shortlisting/"
)
# JSON format (step 2 results)
traj_dir = "/Users/anu/Documents/GitHub/routing/main/tool-response-reflection/Shortlisting/traj/"
results_dir = f"/Users/anu/Documents/GitHub/routing/main/tool-response-reflection/{expt}_results_igf"

############
os.makedirs(results_dir, exist_ok=True)

final_data: Dict[str, Any] = {}

with open(data_points_ignored_file, "r") as f:
    to_be_ignored = json.load(f)
win_rate_output = {}

# per model
for model in models:
    data_dir = f"{traj_dir}{model}"
    initial_results_dir = f"{agent_results_dir}{model}"
    files_in_data = os.listdir(data_dir)
    files_in_initial_results_dir = os.listdir(initial_results_dir)

    results = []

    final_data[model] = {"datasets": [], "total_micro": {}, "total_macro": {}}
    loops = {}

    # per file / per dataset
    for item in files_in_data:
        print(item)
        # runs through only 1 time
        for item1 in files_in_initial_results_dir:
            if item != item1:
                continue
            print("item1.....", item1)
            with open(os.path.join(os.path.join(initial_results_dir, item1)), "r") as f:
                data_initial = json.load(f)

            look_for = item1.split("_rest", 1)[0]
            points_to_be_ignored = []
            for k, ignore_l in to_be_ignored.items():
                if look_for in k:
                    points_to_be_ignored = ignore_l
            print(look_for, points_to_be_ignored)
            loops_per_datapoint = []
            samples_per_datapoint_initial_result = []
            for r in data_initial:
                if r["metadata"]["ignore"] is True:
                    continue
                if r["metadata"]["sample_id"] in points_to_be_ignored:
                    continue

                loops_per_datapoint.append(len(r["tao_loop"]))
                samples_per_datapoint_initial_result.append(r["metadata"]["ignore"])

            assert len(loops_per_datapoint) == len(samples_per_datapoint_initial_result)

        loops[item] = loops_per_datapoint

        look_for = item.split("_rest", 1)[0]
        points_to_be_ignored = []
        for k, ignore_l in to_be_ignored.items():
            if look_for in k:
                points_to_be_ignored = ignore_l
        print(look_for, points_to_be_ignored)

        if "DS_Store" in item:
            continue

        with open(os.path.join(data_dir, item), "r") as f:
            data = json.load(f)

        count = 0
        all_points = 0
        wins_list = []
        sample_ids_list = []
        for record in data:
            w = 0
            if record["metadata"]["sample_id"] in points_to_be_ignored:
                continue
            if record["metadata"]["ignore"]:
                continue
            if record["gold_answer"] == record["predicted_answer"]:
                count += 1
                w = 1

            sample_ids_list.append(record["metadata"]["sample_id"])
            wins_list.append(w)
            all_points = all_points + 1

        assert len(sample_ids_list) == len(wins_list)

        datapoints = all_points
        matches_count = count
        win_rate = round(count / datapoints, 2)

        n_loops = []
        for k, val in loops.items():
            if item in k:
                print("FOUND", val)
                n_loops = val
                break

        final_data[model]["datasets"].append(
            {
                "data_file": item,
                "n_datapoints": all_points,
                "n_wins": count,
                "wins": wins_list,
                "sample_ids": sample_ids_list,
                "n_loops": sum([x for x in n_loops]),
            }
        )

        # Append to results list
        results.append([item, datapoints, matches_count, win_rate])

    with open(results_dir + "/" + expt + f"_{model}_loops.json", "w") as f:
        json.dump(loops, f)

    dp_all = 0
    wins_all = 0
    for dd in final_data[model]["datasets"]:
        wins_all = wins_all + dd["n_wins"]
        dp_all = dp_all + dd["n_datapoints"]

    num_total = sum(d["n_datapoints"] for d in final_data[model]["datasets"])

    total_loops = sum(d["n_loops"] for d in final_data[model]["datasets"])
    final_data[model]["total_micro"] = {
        "n_datapoints": dp_all,
        "total_loops": total_loops,
        "all_wins": wins_all,
        "n_datasets": len(final_data[model]["datasets"]),
        "win_rate": sum(d["n_wins"] for d in final_data[model]["datasets"]) / num_total,
        "n_loops": sum(d["n_loops"] for d in final_data[model]["datasets"]) / num_total,
    }

    final_data[model]["total_macro"] = {
        "n_datapoints": dp_all,
        "total_loops": total_loops,
        "all_wins": wins_all,
        "win_rate": sum(
            d["n_wins"] / d["n_datapoints"] for d in final_data[model]["datasets"]
        )
        / len(final_data[model]["datasets"]),
        "n_datasets": len(final_data[model]["datasets"]),
        "n_loops": sum(d["n_loops"] for d in final_data[model]["datasets"]) / num_total,
    }
    win_rate_output[model] = final_data

    # print(json.dumps(win_rate_output, indent=4))
    # Print as a table using tabulate
    # print(
    #     tabulate(
    #         results,
    #         headers=["Filename", "Datapoints", "Matches", "Win Rate"],
    #         tablefmt="csv",
    #     )
    # )


print(json.dumps(final_data, indent=4))

os.makedirs(results_dir, exist_ok=True)
with open(results_dir + "/" + expt + "_win_rate.json", "w") as f:
    json.dump(final_data, f)
