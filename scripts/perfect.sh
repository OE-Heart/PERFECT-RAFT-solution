for task in ade_corpus_v2 banking_77 neurips_impact_statement_risks one_stop_english overruling terms_of_service tweet_eval_hate twitter_complaints
do
    CUDA_VISIBLE_DEVICES=0 python src/main.py configs/$task.json
done

# for task in ade_corpus_v2 terms_of_service tai_safety_research neurips_impact_statement_risks overruling systematic_review_inclusion one_stop_english tweet_eval_hate twitter_complaints semiconductor_org_types
# do
#     CUDA_VISIBLE_DEVICES=0 python src/main.py configs/$task.json
# done