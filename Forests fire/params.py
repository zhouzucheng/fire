
import os


# 项目文件设置
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE_PATH = os.path.join(PROJECT_DIR, 'data/train.csv')
TEST_FILE_PATH = os.path.join(PROJECT_DIR, 'data/test.csv')

# 预处理设置
NUM_WORDS = 5500
PAD = '<PAD>'
PAD_NO = 0
UNK = '<UNK>'
UNK_NO = 1
START_NO = UNK_NO + 1
SENT_LENGTH = 300

# 模型参数
EMBEDDING_SIZE = 300
TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
LEARNING_RATE = 0.0001
EPOCHS = 100
