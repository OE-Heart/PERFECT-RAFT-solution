import abc
from collections import OrderedDict
from os.path import join
from datasets import load_dataset, concatenate_datasets
import functools
import numpy as np
import sys
import torch
from collections import Counter

from metrics import metrics


class RAFT(abc.ABC):
    task = NotImplemented
    num_labels = NotImplemented
    metric = [metrics.accuracy]

    def __init__(self, data_seed, cache_dir, data_dir=None):
        self.data_seed = data_seed
        self.data_dir = data_dir
        self.cache_dir = cache_dir

    def load_datasets(self):
        print("task ", self.task)
        return load_dataset("ought/raft", name=self.task, cache_dir=self.cache_dir)

    def split_datasets(self, datasets):
        shuffled_train = datasets["train"].shuffle(seed=self.data_seed)
        datasets["train"] = shuffled_train.select([i for i in range(25)])
        datasets["validation"] = shuffled_train.select([i for i in range(25, 50)])
        return datasets

    def get_datasets(self):
        datasets = self.load_datasets()
        datasets = self.split_datasets(datasets)
        label_distribution_train = Counter(datasets["train"]["Label"])
        label_distribution_dev = Counter(datasets["validation"]["Label"])
        return datasets


class ade_corpus_v2(RAFT):
    task = "ade_corpus_v2"
    num_labels = 2


class banking_77(RAFT):
    task = "banking_77"
    num_labels = 77


class neurips_impact_statement_risks(RAFT):
    task = "neurips_impact_statement_risks"
    num_labels = 2


class one_stop_english(RAFT):
    task = "one_stop_english"
    num_labels = 3


class overruling(RAFT):
    task = "overruling"
    num_labels = 2


class semiconductor_org_types(RAFT):
    task = "semiconductor_org_types"
    num_labels = 2


class systematic_review_inclusion(RAFT):
    task = "systematic_review_inclusion"
    num_labels = 2


class tai_safety_research(RAFT):
    task = "tai_safety_research"
    num_labels = 2


class terms_of_service(RAFT):
    task = "terms_of_service"
    num_labels = 2


class tweet_eval_hate(RAFT):
    task = "tweet_eval_hate"
    num_labels = 2


class twitter_complaints(RAFT):
    task = "twitter_complaints"
    num_labels = 2


TASK_MAPPING = OrderedDict(
    [
        ("ade_corpus_v2", ade_corpus_v2),
        ("banking_77", banking_77),
        ("neurips_impact_statement_risks", neurips_impact_statement_risks),
        ("one_stop_english", one_stop_english),
        ("overruling", overruling),
        ("semiconductor_org_types", semiconductor_org_types),
        ("systematic_review_inclusion", systematic_review_inclusion),
        ("tai_safety_research", tai_safety_research),
        ("terms_of_service", terms_of_service),
        ("tweet_eval_hate", tweet_eval_hate),
        ("twitter_complaints", twitter_complaints),
    ]
)


class AutoTask:
    @classmethod
    def get(self, task, data_seed, cache_dir, data_dir=None):
        if task in TASK_MAPPING:
            return TASK_MAPPING[task](
                data_seed=data_seed,
                cache_dir=cache_dir,
                data_dir=data_dir,
            )
        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )
