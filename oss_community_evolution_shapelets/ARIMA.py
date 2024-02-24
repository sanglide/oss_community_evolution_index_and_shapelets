# Import libraries
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics import tsaplots
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as plt
# import seaborn as sns
import pandas as pd
import glob
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, load_robot_execution_failures
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, roc_curve


class ARIMA:

    def __init__(self):
        # Download data
        self.__df = None
        # self.__df.set_index(index, inplace=True)
        self.__index_diff_values = None
        self.__in_sample_predicted_values = None
        # self.model=None
        self.__df_origin=None
    def set_attribute(self,df,df_origin):
        # Download data
        self.__df = df
        self.__df_origin = df_origin
        # self.__df.set_index(index, inplace=True)
        self.__index_diff_values = None
        self.__in_sample_predicted_values = None
        # self.model = None
    def __check_stat(self,value,file_preix):
        # Plot original data
        self.__df[value].plot(legend=False, title='Original Data')
        pyplot.savefig(f"./ARIMA/{file_preix}-original-data.png")

        # Apply Augmented Dickey-Fuller Test to original data
        adf_result_before_differencing = adfuller(self.__df[value])
        print('Results of ADF test for original data:')
        print('ADF Statistic: %f', adf_result_before_differencing[0])
        print('p-value: %f', adf_result_before_differencing[1])
        if adf_result_before_differencing[1] >= 0.05:
            print('Fail to reject the null hypothesis (H0) at 5 % level of \
            significance. The data has a unit root and is non-stationary')
        else:
            print('Reject the null hypothesis (H0) at 5 % level of \
            significance. The data does not have a unit root and is stationary')

    def __make_stat(self,value,file_preix):
        #  Differencing to make the series stationary
        va=self.__df[value]
        self.__index_diff_values = self.__df[value] - \
                                   self.__df[value].shift()
        # self.__index_diff_values.dropna(inplace=True)
        vv=va.iloc[0]
        self.__index_diff_values.iloc[0]=0

        # Check stationarity
        # Plot differenced data
        self.__index_diff_values.plot(legend=False, title='Differenced data',
                                      color='green')
        pyplot.savefig(f"./ARIMA/{file_preix}-differenced-data.png")
        pyplot.close('all')


        # Apply Augmented Dickey-Fuller Test after differencing
        adf_result_after_differencing = adfuller(self.__index_diff_values)
        print('Results of ADF test after differencing:')
        print('ADF Statistic: %f', adf_result_after_differencing[0])
        print('p-value: %f', adf_result_after_differencing[1])
        if adf_result_after_differencing[1] >= 0.05:
            print('Fail to reject the null hypothesis (H0) at 5 % level of \
            significance. The data has a unit root and is non-stationary')
        else:
            print('Reject the null hypothesis (H0) at 5 % level of \
            significance. The data does not have a unit root and is stationary')

    def __plot_acf_pacf(self,value,file_preix):
        # Plot ACF
        tsaplots.plot_acf(self.__index_diff_values)
        # tsaplots.plot_acf(self.__index_diff_values, lags=50)
        pyplot.savefig(f"./ARIMA/{file_preix}-acf-data.png")

        # Plot PACF
        tsaplots.plot_pacf(self.__index_diff_values)
        # tsaplots.plot_pacf(self.__index_diff_values, lags=50)
        pyplot.savefig(f"./ARIMA/{file_preix}-pac-data.png")
        pyplot.close('all')

    # def __chekc_forcast(self,value,file_preix):
    #     # Implement ARIMA(3,0,2)
    #     self.model = sm.tsa.arima.ARIMA(self.__index_diff_values.values, order=(1,1,1))
    #     self.model = self.model.fit()
    #     print(self.model.summary())
    #
    #     # In-sampele forecast
    #     self.__in_sample_predicted_values = self.model.fittedvalues.cumsum() + \
    #                                         self.__df[value].iloc[0]
    #
    #     self.__in_sample_predicted_values = [self.__df[value].iloc[0]] + list(
    #         self.__in_sample_predicted_values)
    #
    #     self.__df['predicted'] = self.__in_sample_predicted_values
    #
    #     self.__df.to_csv(f"./ARIMA/{file_preix}.csv")
    #
    #     # self.__df.columns = ['actual', 'predicted']
    #     df_plot=self.__df[[value, 'predicted']]
    #
    #     df_plot.plot()
    #     pyplot.savefig(f"./ARIMA/{file_preix}-prediction-data.png")
    #     pyplot.close('all')
    # def multiple_forecast(self,value,file_preix,step=48):
    #     # df=pd.Series(self.__df[value])
    #     # for step in range(48):
    #     #     forecast=self.model.forecast(steps=1)
    #     #     df = df.append(pd.DataFrame({'commit_count': forecast}), ignore_index=True)
    #     valuee=list(self.model.forecast(steps=step))
    #     value_before=list(self.__df[value])
    #     value_before.extend(valuee)
    #
    #     self.__df_origin['predicted_multi'] = value_before
    #
    #     df_plott = self.__df_origin[[value, 'predicted_multi']]
    #
    #     df_plott.plot()
    #     pyplot.savefig(f"./ARIMA/{file_preix}-prediction-data2.png")
    #     pyplot.close('all')
    #
    #     return valuee
    def multiple_prediction(self,value,file_preix,step=60):
        # Implement ARIMA(3,0,2)
        is_column_all_zeros = self.__df["commit_count"].eq(0).all()

        if is_column_all_zeros:
            prediction_value=np.array([0 for i in range(600)])
        else:

            model = sm.tsa.arima.ARIMA(self.__index_diff_values.values, order=(1, 1, 1))
            model = model.fit()
            print(model.summary())

            prediction_diff=list(model.forecast(steps=step))
            prediction_value = np.array(prediction_diff).cumsum()+self.__df[value].iloc[0]
            prediction_value = np.where(prediction_value < 0, 0, prediction_value)

        value_before = list(self.__df[value])
        value_before.extend(prediction_value[:(len(self.__df_origin)-len(value_before))])

        self.__df_origin['predicted_multi'] = value_before
        self.__df_origin.to_csv(f"./ARIMA/{file_preix}.csv")

        df_plott = self.__df_origin[[value, 'predicted_multi']]

        df_plott.plot()
        pyplot.axvline(x=self.__df_origin.iloc[len(self.__df)-1].name, ls="-.", c="red")  # 添加垂直直线
        pyplot.savefig(f"./ARIMA/{file_preix}-prediction-data2.png")
        pyplot.close('all')

    def get_differencies(self):
        return self.__index_diff_values

    def get_predicted(self):
        return self.__in_sample_predicted_values

    def get_actual(self):
        return self.__df

    def main(self,value,file_preix,step):
        is_column_all_zeros = self.__df[value].eq(0).all()

        if is_column_all_zeros:
            self.multiple_prediction(value,file_preix,step)
        else:
            self.__check_stat(value,file_preix)
            self.__make_stat(value,file_preix)
            self.__plot_acf_pacf(value,file_preix)
            # self.__chekc_forcast(value,file_preix)
            # self.get_multiple_step_forecast(value, file_preix, 10)
            self.multiple_prediction(value,file_preix,step)

def prediction_preprocess(directory,data_period_month,forecast_gap_month,label_period_month):
    # 1. 生成一个新的index_productivity.csv

    # Specify the directory you want to use

    # Get a list of all CSV files in the directory
    csv_files = glob.glob(directory + '*.csv')

    # Initialize an empty list to hold dataframes
    df_list = []
    label_true=[]
    label_test=[]
    repo_list=[]

    # Loop through the list of CSV files
    for filename in csv_files:
        if "index_productivity" in filename:
            continue
        # Read each file into a dataframe and append to the list
        repo_list.append(filename[:len(filename)-4].replace("__","/"))

        df_repo=pd.read_csv(filename)

        last_non_zero_index = df_repo[df_repo["commit_count"] != 0].last_valid_index()


        # start_d,end_d=last_non_zero_index-label_period_month-forecast_gap_month-data_period_month,\
        #               last_non_zero_index-label_period_month-forecast_gap_month
        # start_l,end_l=len(df_repo)-label_period_month,len(df_repo)

        if len(df_repo) - label_period_month - forecast_gap_month < last_non_zero_index:
            start_d, end_d = len(df_repo) - label_period_month - forecast_gap_month - data_period_month, \
                             len(df_repo) - label_period_month - forecast_gap_month
            start_l, end_l = len(df_repo) - label_period_month+1, len(df_repo)
        else:
            start_d, end_d = last_non_zero_index - forecast_gap_month - data_period_month, \
                             last_non_zero_index - forecast_gap_month
            start_l, end_l = last_non_zero_index+1, last_non_zero_index + label_period_month+1
            start_l, end_l = last_non_zero_index, last_non_zero_index + label_period_month

        df_repo0=df_repo.iloc[start_d:end_d,]
        df_label=df_repo.iloc[start_l:end_l,]
        df_label=df_label.reset_index()
        df_list.append(df_repo0[['time','project_name','predicted_multi']])

        l=0
        count = 0
        for i in df_label["commit_count"]:
            if float(i) > 0:
                count = count + 1
            if count > 0:
                l = 1
                break
        ll = 0
        count=0
        for i in df_label["predicted_multi"]:
            if float(i) > 1:
                count=count+1
            if count>5:
                ll = 1
                break
        label_true.append(l)
        label_test.append(ll)

    print(f"======================= ARIMA prediction {len(label_true)} ===============================")
    print(label_true)
    print(label_test)
    print(precision_score(label_true,label_test))
    print(confusion_matrix(label_true,label_test))
    print(classification_report(label_true,label_test))

    # Concatenate all the dataframes in the list
    result = pd.concat(df_list, ignore_index=True)

    # Write the concatenated dataframe to a new CSV file
    result.to_csv(directory + 'index_productivity_predictioned.csv', index=False)

    # 2. 切割数据、判断标签



    # 3. return df、label
    return result,df_label


def prediction_two_classification(df,label):
    #todo: [time,project_name,predicted]
    extraction_settings = ComprehensiveFCParameters()
    # column_id (str) – The name of the id column to group by
    # column_sort (str) – The name of the sort column.
    X = extract_features(df,
                         column_id='project_name', column_sort='time',
                         default_fc_parameters=extraction_settings,
                         impute_function=impute)
    X.head()
    X_filtered = extract_relevant_features(df, label,
                                           column_id='project_name', column_sort='time',
                                           default_fc_parameters=extraction_settings)
    X_filtered.head()
    X_train, X_test, X_filtered_train, X_filtered_test, y_train, y_test = train_test_split(X, X_filtered, label,
                                                                                           test_size=.4)

    cl = DecisionTreeClassifier()
    cl.fit(X_train, y_train)
    print(classification_report(y_test, cl.predict(X_test)))
    # pd.Series(rfr.feature_importances_, index=df.feature_names).sort_values(ascending=False)

if __name__ == "__main__":
    df = pd.read_csv('')
    # df = pd.read_csv('C:\\phd-one\\project\\20240103-tosem-response\\0327_oss_community_evolution_shapelets\\data\\index_productivity_32.csv')
    df_repo_list=list(set(df["project_name"]))
    df_sample=[]
    for i in df_repo_list:
        df_new=df[df["project_name"]==i]
        df_new=df_new.reset_index()
        df_sample.append(df_new)
    print(f'------------- sample shape : {len(df_sample)} --------------')
    count=0

    ARIMA = ARIMA()


    # re=[]
    # re_repo=[]

    for sample in df_sample:
        re_temp=[]
        print(f'=============== project : {df_repo_list[count]} ===============')
        # if df_repo_list[count]=="thoughtbot/paperclip":
        if True:
            # re_repo.append(df_repo_list[count])

            label_period_month = 12 * 4
            # forecast_gap_month_list = [3 * 4,24,36,48,60,72,84,96]
            forecast_gap_month_list = [3 * 4]
            data_period_month = 12 * 4
            last_non_zero_index = sample[sample["commit_count"] != 0].last_valid_index()

            for forecast_gap_month in forecast_gap_month_list:
                if len(sample)-label_period_month-forecast_gap_month<last_non_zero_index:
                    start_d, end_d = len(sample) -label_period_month- forecast_gap_month - data_period_month, \
                                     len(sample) -label_period_month- forecast_gap_month
                    start_l, end_l = len(sample) - label_period_month, len(sample)
                else:
                    start_d, end_d = last_non_zero_index -  forecast_gap_month - data_period_month, \
                                     last_non_zero_index -  forecast_gap_month
                    start_l, end_l = last_non_zero_index, last_non_zero_index+ label_period_month
                # step=len(sample)-end_d
                step=600
                # start_l, end_l = len(sample) - label_period_month, len(sample)
                df_repo0 = sample.iloc[start_d:end_d, ]
                df_label = sample.iloc[start_l:end_l, ]

                df_repo0 = df_repo0.reset_index()
                df_origin=sample.iloc[start_d:end_l, ]
                df_origin.reset_index()

                # re_temp.append(len(df_repo0))



                ARIMA.set_attribute(df_repo0,df_origin)
                # ARIMA.set_attribute(sample,sample)

                ARIMA.main("commit_count",df_repo_list[count].replace("/","__"),step)
                # ARIMA.multiple_forecast("commit_count",df_repo_list[count].replace("/","-"),step=48)

                # a = ARIMA.get_actual()
                #
                # b = ARIMA.get_predicted()


            # re.append(re_temp)

            # dffff = pd.DataFrame(re, index=re_repo)
            # dffff.to_csv("length.csv")

        count=count+1

    label_period_month = 12 * 4
    # forecast_gap_month_list = [3 * 4,24,36,48,60,72,84,96]
    forecast_gap_month = 3 * 4
    data_period_month = 12 * 4
    result,label=prediction_preprocess("./ARIMA/",data_period_month,forecast_gap_month,label_period_month)
