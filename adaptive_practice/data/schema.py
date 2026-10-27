TRAIN_DTYPES = {
    "row_id": "int64",
    "timestamp": "int64",
    "user_id": "int32",
    "content_id": "int16",
    "content_type_id": "int8",
    "task_container_id": "int16",
    "user_answer": "int8",
    "answered_correctly": "int8",
    "prior_question_elapsed_time": "float32",
    "prior_question_had_explanation": "boolean",
}

QUESTION_COLUMNS = ["question_id", "bundle_id", "correct_answer", "part", "tags"]
LECTURE_COLUMNS = ["lecture_id", "tag", "part", "type_of"]

QUESTION = 0
LECTURE = 1

N_PARTS = 7
N_TAGS = 188
SESSION_GAP_MS = 30 * 60 * 1000
