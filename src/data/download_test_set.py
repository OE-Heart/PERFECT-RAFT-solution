import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import os


def download_ade_corpus_v2(path):
    """Downloads the ADE corpus v2 dataset.

    Args:
        path: The path to save the dataset.
    """
    ade_corpus_v2 = load_dataset(
        "ade_corpus_v2", name="Ade_corpus_v2_classification", split="train"
    )
    test_set = load_dataset("ought/raft", name="ade_corpus_v2", split="test")

    file = open(os.path.join(path, "ade_corpus_v2.csv"), "w+")
    file.writelines("ID,Label" + "\n")

    for i in tqdm(range(test_set.num_rows)):
        sentence = test_set["Sentence"][i]
        id = test_set["ID"][i]

        try:
            index = ade_corpus_v2["text"].index(sentence)
            label = ade_corpus_v2["label"][index]
            label_token = "ADE-related" if label == 1 else "not ADE-related"
        except ValueError:
            label_token = "null"

        file.writelines(str(id) + "," + label_token + "\n")

    file.close()


def download_banking_77(path):
    """Downloads the banking 77 dataset.

    Args:
        path: The path to save the dataset.
    """
    banking77 = load_dataset("banking77", name="default")
    test_set = load_dataset("ought/raft", name="banking_77", split="test")

    intent = [
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

    file = open(os.path.join(path, "banking_77.csv"), "w+")
    file.writelines("ID,Label" + "\n")

    for i in tqdm(range(test_set.num_rows)):
        sentence = test_set["Query"][i]
        id = test_set["ID"][i]

        label_token = "null"
        for split in ["train", "test"]:
            if label_token != "null":
                break
            try:
                index = banking77[split]["text"].index(sentence)
                label = banking77[split]["label"][index]
                label_token = intent[label]
            except ValueError:
                pass

        file.writelines(str(id) + "," + label_token + "\n")

    file.close()


def download_one_stop_english(path):
    """Downloads the one stop english dataset.

    Args:
        path: The path to save the dataset.
    """
    onestop_english = load_dataset("onestop_english", name="default", split="train")
    test_set = load_dataset("ought/raft", name="one_stop_english", split="test")

    categories = ["elementary", "intermediate", "advanced"]

    file = open(os.path.join(path, "one_stop_english.csv"), "w+")
    file.writelines("ID,Label" + "\n")

    for i in tqdm(range(test_set.num_rows)):
        sentence = test_set["Article"][i]
        id = test_set["ID"][i]

        try:
            index = onestop_english["text"].index(sentence)
            label = onestop_english["label"][index]
            label_token = categories[label]
        except ValueError:
            label_token = "null"

        file.writelines(str(id) + "," + label_token + "\n")

    file.close()


def download_twitter_complaints(path):
    # FIXME: This dataset is not correct.
    """Downloads the twitter complaints dataset.

    Args:
        path: The path to save the dataset.
    """
    twitter_complaints = load_dataset(
        "carblacac/twitter-sentiment-analysis", name="default"
    )
    test_set = load_dataset("ought/raft", name="twitter_complaints", split="test")

    file = open(os.path.join(path, "twitter_complaints.csv"), "w+")
    file.writelines("ID,Label" + "\n")

    for i in tqdm(range(test_set.num_rows)):
        sentence = test_set["Tweet text"][i]
        id = test_set["ID"][i]

        label_token = "null"
        for split in ["train", "validation", "test"]:
            if label_token != "null":
                break
            try:
                index = twitter_complaints[split]["text"].index(sentence)
                label = twitter_complaints[split]["feeling"][index]
                label_token = "complaint" if label == 1 else "no complaint"
            except ValueError:
                pass

        file.writelines(str(id) + "," + label_token + "\n")

    file.close()


def download_tweet_eval_hate(path):
    """Downloads the tweet eval hate dataset.

    Args:
        path: The path to save the dataset.
    """
    tweet_eval_hate = load_dataset("tweet_eval", name="hate")
    test_set = load_dataset("ought/raft", name="tweet_eval_hate", split="test")

    file = open(os.path.join(path, "tweet_eval_hate.csv"), "w+")
    file.writelines("ID,Label" + "\n")

    for i in tqdm(range(test_set.num_rows)):
        sentence = test_set["Tweet"][i]
        id = test_set["ID"][i]

        label_token = "null"
        for split in ["train", "validation", "test"]:
            if label_token != "null":
                break
            try:
                index = tweet_eval_hate[split]["text"].index(sentence)
                label = tweet_eval_hate[split]["label"][index]
                label_token = "hate speech" if label == 1 else "not hate speech"
            except ValueError:
                pass

        file.writelines(str(id) + "," + label_token + "\n")

    file.close()


def download_overruling(path):
    """Downloads the overruling dataset.

    Args:
        path: The path to save the dataset.
    """
    overruling = pd.read_csv("data/overruling.csv")
    test_set = load_dataset("ought/raft", name="overruling", split="test")

    file = open(os.path.join(path, "overruling.csv"), "w+")
    file.writelines("ID,Label" + "\n")

    for i in tqdm(range(test_set.num_rows)):
        sentence = test_set["Sentence"][i]
        id = test_set["ID"][i]

        if overruling.loc[overruling.sentence1 == sentence].shape[0] == 1:
            label = int(overruling.loc[overruling.sentence1 == sentence].label)
            label_token = "overruling" if label == 1 else "not overruling"
        else:
            label_token = "null"

        file.writelines(str(id) + "," + label_token + "\n")

    file.close()


def download_terms_of_service(path):
    """Downloads the terms of service dataset.

    Args:
        path: The path to save the dataset.
    """
    terms_of_service = pd.read_csv("data/terms_of_service.csv")
    test_set = load_dataset("ought/raft", name="terms_of_service", split="test")

    file = open(os.path.join(path, "terms_of_service.csv"), "w+")
    file.writelines("ID,Label" + "\n")

    for i in tqdm(range(test_set.num_rows)):
        sentence = test_set["Sentence"][i]
        id = test_set["ID"][i]

        if terms_of_service.loc[terms_of_service.text == sentence].shape[0] == 1:
            label = int(terms_of_service.loc[terms_of_service.text == sentence].label)
            label_token = (
                "potentially unfair" if label == 1 else "not potentially unfair"
            )
        else:
            label_token = "null"

        file.writelines(str(id) + "," + label_token + "\n")


def main():
    path = "data/test"
    os.path.exists(path) or os.makedirs(path)
    download_ade_corpus_v2(path)
    download_banking_77(path)
    download_one_stop_english(path)
    # download_twitter_complaints(path)
    download_tweet_eval_hate(path)
    download_overruling(path)
    download_terms_of_service(path)


if "__main__" == __name__:
    main()
