from load_data import Data
import logging
from train import train

n_q = 123  

dat = Data(n_question=n_q, seqlen=200, separate_char=',') 

train_data_path = "../data/2009_skill_builder_data_corrected/assist2009_train.txt"
test_data_path = "../data/2009_skill_builder_data_corrected/assist2009_test.txt"

# dataformat: (q_data_array, qa_data_array)
train_data = dat.load_data(train_data_path)
test_data = dat.load_data(test_data_path)

# 设置日志记录
logging.basicConfig(filename='training_dkvmn.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

train(train_data, test_data)
