import global_settings
import datetime

from main_calculate_indexes import execute_all


def canshu_exp():

    CNM_RESOLUTION=[0.1,0.3,0.5,0.7,0.9,1,2,3,4,5,6,10,100]

    for i_sha in CNM_RESOLUTION:
        global_settings.CNM_RESOLUTION=i_sha
        print(f'----------------------------- CNM_RESOLUTION = {global_settings.CNM_RESOLUTION} --------------------------------')

        # window_size_experiment()
        execute_all()
        f = open("./terminated", "w")
        f.write(str(datetime.datetime.now()))
        f.close()

if __name__=="__main__":
    canshu_exp()