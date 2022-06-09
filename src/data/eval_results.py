import sys
import pandas as pd
import os


def eval(dataset_name, results_path):
    eval_path = "data/test/" + dataset_name + ".csv"
    # result_path = "results/" + dataset_name + ".csv"
    result_path = os.path.join(results_path, dataset_name, "predictions.csv")
    if not os.path.exists(eval_path) or not os.path.exists(result_path):
        return 0
    eval = pd.read_csv(eval_path)
    result = pd.read_csv(result_path)

    df = eval.dropna().join(result, lsuffix=("_eval"))
    accuracy = (df.Label == df.Label_eval).value_counts()[True] / df.shape[0]

    return accuracy


def main():
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        results_path = str(input("Results path: "))
    dataset_list = [
        "ade_corpus_v2",
        "banking_77",
        "neurips_impact_statement_risks",
        "one_stop_english",
        "overruling",
        "semiconductor_org_types",
        "systematic_review_inclusion",
        "tai_safety_research",
        "terms_of_service",
        "tweet_eval_hate",
        "twitter_complaints",
    ]
    accuracy_list = {}
    for dataset_name in dataset_list:
        accuracy_list[dataset_name] = eval(dataset_name, results_path)
        print(dataset_name + " : " + str(accuracy_list[dataset_name]))

    print("overall : " + str(sum(accuracy_list.values()) / len(accuracy_list)))


if __name__ == "__main__":
    main()
