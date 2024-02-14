import sys

# even number
N_SHAPELETS = 40
WINDOW_STEP = 1
N_CHANNELS = 4
N_JOBS = 20

LIST_LABEL_PERIOD_MONTHS = [12]
LIST_FORECAST_GAP_MONTHS = [3, 6, 9, 12, 15, 18, 21, 24]
# LIST_FORECAST_GAP_MONTHS = [3]
# LIST_DATA_PERIOD_MONTHS = [3, 6, 9, 12, 15, 18, 21, 24]
LIST_DATA_PERIOD_MONTHS = [12]
# 以data point为单位
LIST_WINDOW_SIZE = [3, 4, 5, 6, 7]  # , 8, 12, 16, 20, 24]   # , 28, 32, 36, 40, 44, 48]

DO_VIS = False
CLASSIFY_THRESHOLD = 0.5

DO_SMOOTH = True
SMOOTH_METHOD=1
# 1为高斯平滑，2为moving average

GAUSS_SIGMA = 0.5
DO_GLOBAL_SCALE = True
DO_LOCAL_SCALE = False
DO_FEATURE_ENGINEERING = False

DO_REMOVE_ZERO_DIST = False

DO_POSITION_PENALTY = False
# penalty for the earliest window, must > 1
MAX_POSITION_PENALTY_RATE = 2.

USE_ABS_DIST = True

# EVOLUTION_EVENT_COMBINATIONS = [[0], [1], [2], [3], [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [0, 1, 2],
#                                 [0, 1, 3], [0, 2, 3], [1, 2, 3], [0, 1, 2, 3]]
EVOLUTION_EVENT_COMBINATIONS = [[0, 1, 2, 3]]

failProjL = ['facebookincubator_create-react-app', 'jcjohnson_neural-style', 'angular-ui_bootstrap',
             'Prinzhorn_skrollr', 'davezuko_react-redux-starter-kit', 'eczarny_spectacle', 'boltdb_bolt',
             'jessesquires_JSQMessagesViewController', 'kevinzhow_PNChart', 'JakeWharton_ActionBarSherlock',
             'onevcat_VVDocumenter-Xcode', 'Compass_compass']

# 'thoughtbot_paperclip', is deprecated

DO_FIX_MINING_PARAMETERS = True
FIXED_MINING_FORECAST_GAP = 3 # months
FIXED_MINING_DATA_PERIOD = 12