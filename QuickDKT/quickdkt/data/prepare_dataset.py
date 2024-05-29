import numpy as np
import tqdm
import pandas as pd
import random

data = pd.read_csv(
    '../../data/2009_skill_builder_data_corrected/skill_builder_data_corrected.csv', 
    encoding = 'ISO-8859-15',
    usecols=['order_id', 'user_id', 'sequence_id','skill_id', 'correct']
    ).dropna(subset=['skill_id'])

raw_question = data.skill_id.unique().tolist()

num_skill = len(raw_question)

questions = { p: i+1 for i, p in enumerate(raw_question) }

def parse_all_seq(students):
    all_sequences = []
    for student_id in students:
        one_student = parse_one_seq(data[data.user_id == student_id])
        all_sequences.extend([one_student])
    return all_sequences

def parse_one_seq(student):
    seq = student.sort_values('order_id')
    q = [questions[q] for q in seq.skill_id.tolist()]
    a = seq.correct.tolist()
    return q, a

students = data.user_id.unique()
sequences = parse_all_seq(students)

def train_test_split(data, train_size=0.8, shuffle=True):
    if shuffle:
        random.shuffle(data)
    boundary_train = round(len(data) * train_size)
    return data[: boundary_train], data[boundary_train: ]

train_sequences, test_sequences = train_test_split(sequences)

def sequences2tl(sequences, trgpath):
    with open(trgpath, 'a', encoding='utf8') as f:
        for seq in tqdm.tqdm(sequences, 'write into file: '):
            questions, answers = seq
            seq_len = len(questions)
            f.write(str(seq_len) + '\n')
            f.write(','.join([str(q) for q in questions]) + '\n')
            f.write(','.join([str(a) for a in answers]) + '\n')

# save triple line format for other tasks
sequences2tl(train_sequences, '../../data/2009_skill_builder_data_corrected/assist2009_train.txt')
sequences2tl(test_sequences, '../../data/2009_skill_builder_data_corrected/assist2009_test.txt')

# # CONVERT TO ONE-HOT FORMAT
# MAX_STEP = 50
# NUM_QUESTIONS = num_skill


# def encode_onehot(sequences, max_step, num_questions):
#     result = []

#     for q, a in tqdm.tqdm(sequences, 'convert to one-hot format: '):
#         length =len(q)
#         # append questions' and answers' length to an integer multiple of max_step
#         mod = 0 if length % max_step == 0 else (max_step - length % max_step)
#         onehot = np.zeros(shape=[length + mod, 2 * num_questions])
#         for i, q_id in enumerate(q):
#             index = int(q_id if a[i] > 0 else q_id + num_questions)
#             onehot[i][index] = 1
#         result = np.append(result, onehot)

#     return result.reshape(-1, max_step, 2 * num_questions)

# # reduce the amount of data for example running faster
# percentage = 1
# train_data = encode_onehot(train_sequences[: int(len(train_sequences) * percentage)], MAX_STEP, NUM_QUESTIONS)
# test_data = encode_onehot(test_sequences[: int(len(test_sequences) * percentage)], MAX_STEP, NUM_QUESTIONS)

# # save onehot data
# np.save('../../data/2009_skill_builder_data_corrected/train_data.npy', train_data)
# np.save('../../data/2009_skill_builder_data_corrected/test_data.npy', test_data)