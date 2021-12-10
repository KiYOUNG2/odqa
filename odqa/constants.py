# PREP
question_column_name = "question"
context_column_name = "context"
max_seq_length = 384
doc_stride = 128
pad_to_max_length = True

# POST
version_2_with_negative = True
n_best_size = 10
max_answer_length = 50
null_score_diff_threshold = 1.0
output_dir = "./output"
answer_column_name = "answers"