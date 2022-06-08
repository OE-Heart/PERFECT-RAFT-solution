"""Implements processors to convert examples to input and outputs, this can be
with integrarting patterns/verbalizers for PET or without."""
import abc
import string
from collections import OrderedDict
from unicodedata import name

from .utils import Text, get_verbalization_ids, remove_final_punctuation, lowercase


class AbstractProcessor(abc.ABC):
    def __init__(self, tokenizer, with_pattern, pattern_id=None, mask_position=None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.mask_token_id
        self.mask_token = tokenizer.mask_token
        self.with_pattern = with_pattern
        self.pattern_id = pattern_id
        self.tokenized_verbalizers = None
        self.mask_position = mask_position

    def get_sentence_parts(self, example, mask_length):
        pass

    def get_prompt_parts(self, example, mask_length):
        pass

    def get_verbalizers(self):
        pass

    def get_target(self, example):
        return example["Label"]

    def get_tokenized_verbalizers(self, example=None):
        """If verbalizers are fixed per examples, this returns back a computed tokenized
        verbalizers, but if this is example dependent, it computes the tokenized verbalizers
        per example. In this function, as a default, we compute the static one."""
        if self.tokenized_verbalizers is not None:
            return self.tokenized_verbalizers

        verbalizers = self.get_verbalizers()
        assert (
            len(verbalizers) != 0
        ), "for using static tokenized verbalizers computation, the length"
        "of verbalizers cannot be empty."
        self.tokenized_verbalizers = [
            [get_verbalization_ids(word=verbalizer, tokenizer=self.tokenizer)]
            for verbalizer in verbalizers
        ]
        return self.tokenized_verbalizers

    def get_extra_fields(self, example=None):
        # If there is a need to keep extra information, here we keep a dictionary
        # from keys to their values.
        return {}

    def get_classification_parts(self, example):
        pass

    def get_parts_with_setting_masks(self, part_0, part_1, masks):
        "Only used in case of two sentences: 0`: [p,h,m],[]  `1`: [p,m,h],[]  `2`: [p],[m,h] , `3`: [p],[h,m]"
        if self.mask_position == "0":
            return part_0 + part_1 + masks, []
        elif self.mask_position == "1":
            return part_0 + masks + part_1, []
        elif self.mask_position == "2":
            return part_0, masks + part_1
        elif self.mask_position == "3":
            return part_0, part_1 + masks


class ade_corpus_v2(AbstractProcessor):
    name = "ade_corpus_v2"

    def get_classification_parts(self, example):
        return example["Sentence"], None

    def get_sentence_parts(self, example, mask_length):
        if not self.with_pattern:
            return [Text(text=example["Sentence"], shortenable=True)] + mask_length * [
                Text(text=self.mask_token)
            ], []
        return self.get_prompt_parts(example, mask_length)

    def get_prompt_parts(self, example, mask_length):
        sentence = Text(text=example["Sentence"], shortenable=True)
        masks = mask_length * [Text(text=self.mask_token)]
        return [sentence, Text(text="It's"), *masks, Text(text=".")], []

    def get_verbalizers(self):
        return ["ADE-related", "not ADE-related"]


PROCESSOR_MAPPING = OrderedDict(
    [
        ("ade_corpus_v2", ade_corpus_v2),
        # ("banking_77", banking_77),
        # ("neurips_impact_statement_risks", neurips_impact_statement_risks),
        # ("one_stop_english", one_stop_english),
        # ("overruling", overruling),
        # ("semiconductor_org_types", semiconductor_org_types),
        # ("systematic_review_inclusion", systematic_review_inclusion),
        # ("tai_safety_research", tai_safety_research),
        # ("terms_of_service", terms_of_service),
        # ("tweet_eval_hate", tweet_eval_hate),
        # ("twitter_complaints", twitter_complaints),
    ]
)


class AutoProcessor:
    @classmethod
    def get(self, task, tokenizer, with_pattern, pattern_id, mask_position):
        if task in PROCESSOR_MAPPING:
            return PROCESSOR_MAPPING[task](
                tokenizer=tokenizer,
                with_pattern=with_pattern,
                pattern_id=pattern_id,
                mask_position=mask_position,
            )
        raise ValueError(
            "Unrecognized task {} for AutoProcessor: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in PROCESSOR_MAPPING.keys())
            )
        )
