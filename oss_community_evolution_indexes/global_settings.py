import math

TURN_ON_TEXT_LOG = True
GAMMA = 1 / math.e  # decay
TIME_INTERVAL = 7  # days, i.e., window size
TIME_INTERVALS_IN_SNAPSHOT = 12  # the number of time intervals defined by TIME_INTERNAL days contained in a snapshot

proj_list = ['angular-ui_bootstrap', 'apache_couchdb', 'beautify-web_js-beautify', 'boltdb_bolt',
             'carrierwaveuploader_carrierwave', 'celery_celery', 'Compass_compass', 'davezuko_react-redux-starter-kit',
             'divio_django-cms', 'django-extensions_django-extensions', 'eczarny_spectacle',
             'eventmachine_eventmachine', 'facebookincubator_create-react-app', 'i3_i3',
             'JakeWharton_ActionBarSherlock', 'jcjohnson_neural-style', 'jekyll_jekyll',
             'jessesquires_JSQMessagesViewController', 'jruby_jruby', 'junit-team_junit4', 'kevinzhow_PNChart',
             'mbleigh_acts-as-taggable-on', 'onevcat_VVDocumenter-Xcode', 'prawnpdf_prawn', 'Prinzhorn_skrollr',
             'rails_rails', 'redis_redis-rb', 'sferik_twitter', 'Shopify_liquid', 'sinatra_sinatra',
             'sparklemotion_nokogiri', 'thoughtbot_paperclip', ]

activeProjL = ['sinatra_sinatra', 'sparklemotion_nokogiri', 'i3_i3', 'rails_rails', 'jekyll_jekyll',
               'Shopify_liquid', 'sferik_twitter', 'django-extensions_django-extensions', 'divio_django-cms',
               'carrierwaveuploader_carrierwave', 'prawnpdf_prawn', 'mbleigh_acts-as-taggable-on',
               'beautify-web_js-beautify', 'jruby_jruby', 'eventmachine_eventmachine',
               'redis_redis-rb', 'junit-team_junit4', 'apache_couchdb', 'celery_celery']

failProjL = ['facebookincubator_create-react-app', 'jcjohnson_neural-style', 'angular-ui_bootstrap',
             'Prinzhorn_skrollr', 'davezuko_react-redux-starter-kit', 'eczarny_spectacle', 'boltdb_bolt',
             'jessesquires_JSQMessagesViewController', 'kevinzhow_PNChart', 'JakeWharton_ActionBarSherlock',
             'onevcat_VVDocumenter-Xcode', 'Compass_compass']


EVOLUTION_PATTERN_THRESHOLD = 0.1

SOURCE_DATA_PATH = "./data/chosenProjData.csv"  # data source used for FSE submission
NEW_DATA_DIR = "./data/"

# True: perform community evolution analysis for the two hundred projects in the evaluation set
# False: perform community evolution analysis for the thirty-two projects in the example set
USE_NEW_DATA = False

CORRELATION_METHOD = "spearman"

# 0: math.log(dc[key] + 1, math.e)
# 1: 1
NODE_WEIGHT_FUNC = 0

# 0: math.log(dc[key] + 1, math.e)
# 1: 1
EDGE_WEIGHT_FUNC = 0

#  resolution : float (default=1)
#         If resolution is less than 1, modularity favors larger communities.
#         Greater than 1 favors smaller communities.
CNM_RESOLUTION = 1
