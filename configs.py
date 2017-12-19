import logging
import logging.handlers
import os
from datetime import datetime

MODEL = 'seq2seq'
APP_NAME = os.getenv('APPNAME', MODEL + datetime.now().strftime("%m%d-%H%M"))
QUICK_TEST = False

LOG_FORMAT = '%(asctime)s ' + MODEL + ':%(filename)s:%(funcName)s:[%(levelname)s] %(message)s'
RANDOM_SEED = 531
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'
WORK_DIR = ROOT_DIR if QUICK_TEST else '/opt/workspace/'
DATA_DIR = WORK_DIR + 'data/'
LOG_DIR = WORK_DIR + 'log/'
SCRIPT_DIR = ROOT_DIR + 'script/'

SYNC_SCRIPT_PATH = SCRIPT_DIR + 'sync_s3.sh'
DOWNLOAD_SCRIPT_PATH = SCRIPT_DIR + 'load_s3_json.sh'
LOG_PATH = '%s%s/app.log' % (LOG_DIR, APP_NAME)

Q2A_PATH = DATA_DIR + "query2sku-merged-ad1-20170613+28D+20170710.json.gz"
Q2AD_PATH = DATA_DIR + "query2sku-merged-ad%d-20170613+28D+20170710.json.gz"
Q2A_INFO = DATA_DIR + "query2brand.json"

LOSS_JITTER = 1e-4
SYNC_INTERVAL = 300.0  # sync every 5 minutes
SYNC_TIMEOUT = 600
FIRST_SYNC_DELAY = 300.0  # do the first task only after 5 minutes.

GIT_URL = 'https://github.bus.zalan.do/sl/query2sku'
TB_URL = 'https://query2sku-tensorboard.deepthought.zalan.do'
SUP_URL = 'https://query2sku-supervisor.deepthought.zalan.do'
BK_URL = 'https://query2sku-baker.deepthought.zalan.do'

Q2A_JSON_AKEY1 = 'skus'
Q2A_JSON_AKEY2 = 'id'
