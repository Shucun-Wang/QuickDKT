import pandas as pd
import tqdm

data = pd.read_csv(
    '../../data/anonymized_full_release_competition_dataset/anonymized_full_release_competition_dataset.csv',
    usecols=['startTime', 'timeTaken', 'studentId', 'skill', 'problemId', 'correct']
).dropna(subset=['skill', 'problemId']).sort_values('startTime')

data.timeTaken = data.timeTaken.astype(int)

skills = data.skill.unique().tolist()
problems = data.problemId.unique().tolist()
at = data.timeTaken.unique()

# question id from 1 to #num_skill
skill2id = { p: i+1 for i, p in enumerate(skills) }
problem2id = { p: i+1 for i, p in enumerate(problems) }
at2id = { a: i for i, a in enumerate(at) }

print("number of skills: %d" % len(skills))
print("number of problems: %d" % len(problems))
print("number of answer time: %d" % len(at))

import numpy as np

it = set()
# calculate interval time
for u in data.studentId.unique():
    startTime = np.array(data[data.studentId == u].startTime)
    for i in range(1, len(startTime)):
        item = (startTime[i] - startTime[i - 1]) // 60
        if item > 43200:
            item = 43200
        it.add(item)

it2id = { a: i for i, a in enumerate(it) }
print("number of interval time: %d" % len(it))

# problems to skills
problem2skill = {}
for s, p in zip(np.array(data.skill), np.array(data.problemId)):
    problem2skill[problem2id[p]] = skill2id[s]
with open('../../data/anonymized_full_release_competition_dataset/problem2skill', 'w', encoding='utf-8') as f:
    f.write(str(problem2skill))

def parse_all_seq(students):
    all_sequences = []
    for student_id in tqdm.tqdm(students, 'parse student sequence:\t'):
        student_sequence = parse_student_seq(data[data.studentId == student_id])
        all_sequences.extend([student_sequence])
    return all_sequences

def parse_student_seq(student):
    seq = student
    s = [skill2id[q] for q in seq.skill.tolist()]
    a = seq.correct.tolist()
    p = [problem2id[p] for p in seq.problemId.tolist()]
    it = [0]
    startTime = np.array(seq.startTime)
    for i in range(1, len(startTime)):
        item = (startTime[i] - startTime[i - 1]) // 60
        if item > 43200:
            item = 43200
        it.append(it2id[item])
    at = [at2id[int(x)] for x in seq.timeTaken.tolist()]
    return s, a, p, it, at

sequences = parse_all_seq(data.studentId.unique())

from sklearn.model_selection import train_test_split, KFold

# split train data and test data
train_data, test_data = train_test_split(sequences, test_size=.2, random_state=10)
print(type(train_data))
train_data = np.array(train_data, dtype=object)
test_data = np.array(test_data, dtype=object)

def sequences2l(sequences, trg_path):
    with open(trg_path, 'a', encoding='utf8') as f:
        for seq in tqdm.tqdm(sequences, 'write data into file: %s' % trg_path):
            s_seq, a_seq, p_seq, it_seq, at_seq = seq
            seq_len = len(s_seq)
            f.write(str(seq_len) + '\n')
            f.write(','.join([str(p) for p in p_seq]) + '\n')
            f.write(','.join([str(s) for s in s_seq]) + '\n')
            f.write(','.join([str(a) for a in a_seq]) + '\n')

sequences2l(train_data, '../../data/anonymized_full_release_competition_dataset/assistchall_train_pid.txt')
sequences2l(test_data, '../../data/anonymized_full_release_competition_dataset/assistchall_test_pid.txt')