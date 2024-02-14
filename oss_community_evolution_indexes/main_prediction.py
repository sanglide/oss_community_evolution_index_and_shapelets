import matplotlib.pylab as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

from feature_extraction import extract_first, extract_second
from utilsPY import openreadtxt


def drawMatrix(y_test,y_pred,name,fig_folder):
    print("----------draw {0}-------------".format(name))
    # generate confusion matrix and visualisation
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    # print confusion matrix
    print(confmat)
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.show()
    # 'name' is project name, 'fig_folder' is the path which saves confusion matrix figures.
    plt.savefig(fig_folder+"confusion_matrix_"+name+".jpg", dpi=150)
    # print precision/recall/f1 score
    print('precision:%.3f' % precision_score(y_true=y_test, y_pred=y_pred))
    print('recall:%.3f' % recall_score(y_true=y_test, y_pred=y_pred))
    print('F1:%.3f' % f1_score(y_true=y_test, y_pred=y_pred))

def reportData(x, y, model, model_name, cv):
    # 'cv' is from 'sklearn' package
    # print the model type
    print("\033[0;32;40m------start a {0}------\033[0m".format(model_name))

    pipeline = make_pipeline(StandardScaler(), model)

    # Set the cross-validation fold number cv=10 to use StratifiedKFold with ten folds,
    # and then pass the pipeline and data set to the cross-validation object
    scores = cross_val_score(pipeline, X=x, y=y, cv=10, n_jobs=1, scoring='accuracy')
    print('Cross Validation accuracy scores: %s' % scores)
    print('Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    # Create a StratifiedKFold instance to get the indices of the different training and test samples, with a fold of 10
    strtfdKFold = StratifiedKFold(n_splits=10)
    # Pass features and labels to StratifiedKFold instance
    kfold = strtfdKFold.split(x, y)
    y_pred_sum = []
    y_true_sum = []
    # Loop iteration, (K-1) copies are used for training, 1 copy is used for verification, and the performance of each model is recorded.
    scores = []
    for k, (train, test) in enumerate(kfold):
        pipeline.fit(x.iloc[train, :], y.iloc[train,:])

        y_pred = pipeline.predict(x.iloc[test, :])
        y_pred_sum.extend(y_pred)
        y_true_sum.extend(y.iloc[test,:]["is_success"].tolist())

        score = pipeline.score(x.iloc[test, :], y.iloc[test])
        scores.append(score)
        # print('Fold: %2d, Training/Test Split Distribution: %s, Accuracy: %.3f' % (
        #     k + 1, np.bincount(y.iloc[train]), score))
    if np.mean(scores) > 0.5:
        print(y_true_sum)
        print(y_pred_sum)
        drawMatrix(y_true_sum, y_pred_sum, model_name)
    print('\n\nCross-Validation accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    print("\033[0;32;40m------finish a {0}------\033[0m".format(model_name))

def __train_and_test_predictor(output_dataset_file):
    # TODO: read x,y from 'output_dataset_file'
    # todo：需要把混淆矩阵和原始结果以数据结构返回，再进行可视化（可以直接用latex的代码形式输出）。
    x,y=[],[]

    # perform ten-fold-cross-validation to evaluate the prediction results.
    # 'model' can equal different machine learning classifier.
    model = RandomForestClassifier()
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    # results should at least include: prediction accuracy, confusion matrix
    reportData(x, y, model, "random forest", cv)



def __extract_features(community_evolution_indices):
    # extract features from the series of four indices
    features = []
    data = community_evolution_indices
    bb = extract_first(data)
    if bb == "none":
        return
    split_mean, shrink_mean, merge_mean, expand_mean, split_var, shrink_var, merge_var, expand_var \
        = bb
    aa = extract_second(data)
    if aa == "none":
        return
    split_count, shrink_count, merge_count, expand_count = aa[0]
    split_area, shrink_area, merge_area, expand_area = aa[1]
    features.append([split_mean, shrink_mean, merge_mean, expand_mean,
                     split_var, shrink_var, merge_var, expand_var,
                     split_count, shrink_count, merge_count, expand_count,
                     split_area, shrink_area, merge_area, expand_area])

    return features


def __load_community_evolution_data(community_evolution_folder, project_name, last_commit_time, gap_period_days,
                                    data_period_days):
    # load community evolution data used for feature extraction.
    # Data will be provided as csv files, one for each project.
    # Path of csv file is (note replace / with __ in owner/project)

    evolution_indices_file = f'{community_evolution_folder}{project_name}.csv'

    # sample data in a evolution_indices_file are as follows

    # {community_evolution_folder}/DefinitelyTyped__DefinitelyTyped_indexes.csv
    # time,split,shrink,merge,expand
    # 2012-10-11T23:45:01Z,0.0,0.0,0.0,0.47560088744999474
    # 2012-10-18T23:45:01Z,0.4803005029414881,0.0,0.3170433812293727,0.14318145571559054
    # 2012-10-25T23:45:01Z,0.0,0.0,0.40166777736717313,0.09699484090193346
    # 2012-11-01T23:45:01Z,0.0,0.0,0.0,0.01082871147372703
    # 2012-11-08T23:45:01Z,0.30484475526378757,1.776648360746324e-16,0.0,0.018867731192227746
    # 2012-11-15T23:45:01Z,0.818530796674546,0.0,0.19686818584006321,0.05861052458313234
    # 2012-11-22T23:45:01Z,0.0,0.0,0.0,0.0
    # 2012-11-29T23:45:01Z,0.0,0.0,0.7390660701976273,0.11031685660324005
    # 2012-12-06T23:45:01Z,0.0,0.0,0.0,0.012623270318769144

    # return four data series with respect to split, shrink, merge, and expand
    pass


def __write_feature_label_data(output_dataset_file, project_name, feature_vector, label):
    # write the project_name, feature vector, and the label (active / inactive) into the output_dataset_file
    pass


def __generate_dataset(community_evolution_folder, output_dataset_file, gap_period_days, data_period_days):
    # for each project in the project list = active + inactive projects
    for proj in project_list:
        project_name = proj
        # load community evolution data, specify: project name, last commit time, gap period by days, and data period by days
        # there should be somewhere we record the last commit time for each project, for the sample data, the repoTimeCheck.csv file should work
        community_evolution_indices = \
            __load_community_evolution_data(community_evolution_folder, project_name, last_commit_time, gap_period_days,
                                            data_period_days)
        feature_vector = __extract_features(community_evolution_indices)

        # label can be get from active and inactive project lists
        __write_feature_label_data(output_dataset_file, project_name, feature_vector, label)
    pass


def execute_prediction(result_folder_name):
    # define project list, and other data
    active_project_list = []  # refer to selectedActiveList in sample data
    inactive_project_list = []  # refer to selectedFailedList in sample data

    # generate dataset, specify the input, output, gap period days, and data period days
    __generate_dataset(f'{result_folder_name}/community_evolution/', output_dataset_file, gap_period_days,
                       data_period_days)

    # train and test the prediction model, write results
    __train_and_test_predictor(output_dataset_file)
    pass


if __name__ == '__main__':
    # 2022-06-10T17-09-43Z_interval_7_days_x_12 is sample data
    # the list of projects, and the community evolution indices in the sample data may be different from the real data
    # so do not try too much time to optimize the prediction accuracy on the sample dataset (generally should be effective)
    execute_prediction("2022-06-10T17-09-43Z_interval_7_days_x_12")
