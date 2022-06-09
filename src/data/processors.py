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
        return example["Label"] - 1

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


class banking_77(AbstractProcessor):
    name = "banking_77"

    def get_classification_parts(self, example):
        return example["Query"], None

    def get_sentence_parts(self, example, mask_length):
        if not self.with_pattern:
            return [Text(text=example["Query"], shortenable=True)] + mask_length * [
                Text(text=self.mask_token)
            ], []
        return self.get_prompt_parts(example, mask_length)

    def get_prompt_parts(self, example, mask_length):
        sentence = Text(text=example["Query"], shortenable=True)
        masks = mask_length * [Text(text=self.mask_token)]
        return [sentence, Text(text="It's"), *masks, Text(text=".")], []

    def get_verbalizers(self):
        return [
            "activate_my_card",
            "age_limit",
            "apple_pay_or_google_pay",
            "atm_support",
            "automatic_top_up",
            "balance_not_updated_after_bank_transfer",
            "balance_not_updated_after_cheque_or_cash_deposit",
            "beneficiary_not_allowed",
            "cancel_transfer",
            "card_about_to_expire",
            "card_acceptance",
            "card_arrival",
            "card_delivery_estimate",
            "card_linking",
            "card_not_working",
            "card_payment_fee_charged",
            "card_payment_not_recognised",
            "card_payment_wrong_exchange_rate",
            "card_swallowed",
            "cash_withdrawal_charge",
            "cash_withdrawal_not_recognised",
            "change_pin",
            "compromised_card",
            "contactless_not_working",
            "country_support",
            "declined_card_payment",
            "declined_cash_withdrawal",
            "declined_transfer",
            "direct_debit_payment_not_recognised",
            "disposable_card_limits",
            "edit_personal_details",
            "exchange_charge",
            "exchange_rate",
            "exchange_via_app",
            "extra_charge_on_statement",
            "failed_transfer",
            "fiat_currency_support",
            "get_disposable_virtual_card",
            "get_physical_card",
            "getting_spare_card",
            "getting_virtual_card",
            "lost_or_stolen_card",
            "lost_or_stolen_phone",
            "order_physical_card",
            "passcode_forgotten",
            "pending_card_payment",
            "pending_cash_withdrawal",
            "pending_top_up",
            "pending_transfer",
            "pin_blocked",
            "receiving_money",
            "Refund_not_showing_up",
            "request_refund",
            "reverted_card_payment",
            "supported_cards_and_currencies",
            "terminate_account",
            "top_up_by_bank_transfer_charge",
            "top_up_by_card_charge",
            "top_up_by_cash_or_cheque",
            "top_up_failed",
            "top_up_limits",
            "top_up_reverted",
            "topping_up_by_card",
            "transaction_charged_twice",
            "transfer_fee_charged",
            "transfer_into_account",
            "transfer_not_received_by_recipient",
            "transfer_timing",
            "unable_to_verify_identity",
            "verify_my_identity",
            "verify_source_of_funds",
            "verify_top_up",
            "virtual_card_not_working",
            "visa_or_mastercard",
            "why_verify_identity",
            "wrong_amount_of_cash_received",
            "wrong_exchange_rate_for_cash_withdrawal",
        ]


class neurips_impact_statement_risks(AbstractProcessor):
    name = "neurips_impact_statement_risks"

    def get_classification_parts(self, example):
        return example["Impact statement"], None

    def get_sentence_parts(self, example, mask_length):
        if not self.with_pattern:
            return [
                Text(text=example["Impact statement"], shortenable=True)
            ] + mask_length * [Text(text=self.mask_token)], []
        return self.get_prompt_parts(example, mask_length)

    def get_prompt_parts(self, example, mask_length):
        sentence = Text(text=example["Impact statement"], shortenable=True)
        masks = mask_length * [Text(text=self.mask_token)]
        return [sentence, Text(text="It's"), *masks, Text(text=".")], []

    def get_verbalizers(self):
        return [
            "doesn't mention a harmful application",
            "mentions a harmful application",
        ]


class one_stop_english(AbstractProcessor):
    name = "one_stop_english"

    def get_classification_parts(self, example):
        return example["Article"], None

    def get_sentence_parts(self, example, mask_length):
        if not self.with_pattern:
            return [Text(text=example["Article"], shortenable=True)] + mask_length * [
                Text(text=self.mask_token)
            ], []
        return self.get_prompt_parts(example, mask_length)

    def get_prompt_parts(self, example, mask_length):
        sentence = Text(text=example["Article"], shortenable=True)
        masks = mask_length * [Text(text=self.mask_token)]
        return [sentence, Text(text="It's"), *masks, Text(text=".")], []

    def get_verbalizers(self):
        return ["advanced", "elementary", "intermediate"]


class overruling(AbstractProcessor):
    name = "overruling"

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
        return ["not overruling", "overruling"]


class semiconductor_org_types(AbstractProcessor):
    pass


class systematic_review_inclusion(AbstractProcessor):
    pass


class tai_safety_research(AbstractProcessor):
    pass


class terms_of_service(AbstractProcessor):
    name = "terms_of_service"

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
        return ["potentially unfair", "not potentially unfair"]


class tweet_eval_hate(AbstractProcessor):
    name = "tweet_eval_hate"

    def get_classification_parts(self, example):
        return example["Tweet"], None

    def get_sentence_parts(self, example, mask_length):
        if not self.with_pattern:
            return [Text(text=example["Tweet"], shortenable=True)] + mask_length * [
                Text(text=self.mask_token)
            ], []
        return self.get_prompt_parts(example, mask_length)

    def get_prompt_parts(self, example, mask_length):
        sentence = Text(text=example["Tweet"], shortenable=True)
        masks = mask_length * [Text(text=self.mask_token)]
        return [sentence, Text(text="It's"), *masks, Text(text=".")], []

    def get_verbalizers(self):
        return ["hate speech", "not hate speech"]


class twitter_complaints(AbstractProcessor):
    name = "twitter_complaints"

    def get_classification_parts(self, example):
        return example["Tweet text"], None

    def get_sentence_parts(self, example, mask_length):
        if not self.with_pattern:
            return [
                Text(text=example["Tweet text"], shortenable=True)
            ] + mask_length * [Text(text=self.mask_token)], []
        return self.get_prompt_parts(example, mask_length)

    def get_prompt_parts(self, example, mask_length):
        sentence = Text(text=example["Tweet text"], shortenable=True)
        masks = mask_length * [Text(text=self.mask_token)]
        return [sentence, Text(text="It's"), *masks, Text(text=".")], []

    def get_verbalizers(self):
        return ["complaint", "no complaint"]


PROCESSOR_MAPPING = OrderedDict(
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
