import os.path

import pandas as pd

import global_settings
from global_settings import SOURCE_DATA_PATH


def read_data_from_csv():
    """
    Load issue and pull-request discussion data from a csv file.

    The csv file looks like the follows:

    ,startTime,createUser,commentsUser,proj,year
    0,2012-10-07T13:33:11Z,,set(),angular-ui_bootstrap,2012
    1,2012-10-07T14:51:51Z,,{'ajoslin'},angular-ui_bootstrap,2012
    2,2012-10-07T14:52:56Z,,{'ajoslin'},angular-ui_bootstrap,2012
    ...
    11,2012-10-15T19:50:26Z,max-mykhailenko,{'ajoslin'},angular-ui_bootstrap,2012
    12,2012-10-16T07:23:32Z,max-mykhailenko,"{'max-mykhailenko', 'ajoslin'}",angular-ui_bootstrap,2012
    13,2012-10-16T07:39:05Z,max-mykhailenko,"{'max-mykhailenko', 'ajoslin'}",angular-ui_bootstrap,2012


    Returns
    -------

    pandas data frame

    """
    data = pd.read_csv(SOURCE_DATA_PATH)
    return data


def list_projects(category: str):
    """

    Parameters
    ----------
    category

    `category` can be one of 'all', 'foreign_active', 'china_active', 'foreign_inactive'

    Returns
    -------

    A list of project names, each name contained in this list can later be used to indicate a project in functions
    `load_issue_pr_data_proj()` and `load_commit_data_proj()`

    """

    if category == 'all':
        fo = open('proj_list/704/all_704-8_sbh_select.list', "r")  # all_704-8_sbh_select
        list_projs = fo.read().splitlines()
        return list_projs
    elif category == 'active':
        fo = open('proj_list/704/selectedActiveList.list', "r")
        list_projs = fo.read().splitlines()
        return list_projs
    elif category == 'failed':
        fo = open('proj_list/704/selectedFailedList.list', "r")
        list_projs = fo.read().splitlines()
        return list_projs
    elif category == 'foreign_active':
        pass
    elif category == 'china_active':
        pass
    elif category == 'foreign_inactive':
        pass
    # elif category == 'file_missing':
    #     fo = open('proj_list/704/704_file_missing.list', "r")
    #     list_projs = fo.read().splitlines()
    #     return list_projs
    # elif category == 'all_without_missing':
    #     fo = open('proj_list/704/all_704.list', "r")
    #     list_projs_all = fo.read().splitlines()
    #
    #     fo = open('proj_list/704/704_file_missing.list', "r")
    #     list_projs_miss = fo.read().splitlines()
    #
    #     list_projs = []
    #     for p in list_projs_all:
    #         if not p in list_projs_miss:
    #             list_projs.append(p)
    #
    #     return list_projs

    else:
        print(f"Undefined category {category} when listing projects")
        assert False


def load_issue_pr_data_proj(proj: str):
    """
    Now the csv files are located in a folder whose path is given by some global variable.


    Parameters
    ----------
    proj

    `proj` is the name of the project whose data we want to load.

    Returns
    -------
    Pandas data frame formatted the same to read_data_from_csv(), which
    contains all the issue and pr data of project `proj`

    """

    """
    WARNING:    I have notices some differences between the csv file used in FSE and the new ones:
                
                1) The first column, i.e., the unnamed column, should be a increasing index starting from 0.
                   But in the new csv files, the first column is always 0. The first column is not used in 
                   subsequent calculation steps. But we should fix this issue in the data.
                   * fixed
                   
                2) The column `commentsUser` is a set in the original data, but seems to be a str in the new data.
                   Make sure that the return data are a set of user names for `commentsUser`.
                   * fixed
                   
                3) The column `proj` contains the full name of a repo like 'open-mmlab/mmdetection', which is great.
                   We need to check the code in the project to make sure that we never use the name of the project as
                   a folder name, in which case the '/' in the repo name will mistakenly be interpreted as a directory.
                   The original csv file replaces '/' with '_'. My suggestion is to use '_'.
                   * keep /, fix file name here
    """

    filename = proj.replace("/", "__")
    if os.path.exists(f"{global_settings.NEW_DATA_DIR}issue/{filename}.csv"):
        data_issue = pd.read_csv(f"{global_settings.NEW_DATA_DIR}issue/{filename}.csv",
                                 usecols=["startTime", "createUser", "commentsUser", "proj", "year"])
        data_issue.insert(loc=len(data_issue.columns), column='type', value=['issue' for i in range(data_issue.shape[0])])
    else:
        data_issue=pd.DataFrame()
    if os.path.exists(f"{global_settings.NEW_DATA_DIR}pr/{filename}.csv"):
        data_pr = pd.read_csv(f"{global_settings.NEW_DATA_DIR}pr/{filename}.csv",
                              usecols=["startTime", "createUser", "commentsUser", "proj", "year"])
        data_pr.insert(loc=len(data_pr.columns), column='type', value=['pr' for i in range(data_pr.shape[0])])
    else:
        data_pr=pd.DataFrame()
    return pd.concat([data_issue, data_pr])


def load_issue_data_proj(proj: str):
    filename = proj.replace("/", "__")
    data_issue = pd.read_csv(f"{global_settings.NEW_DATA_DIR}issue/{filename}.csv",
                             usecols=["startTime", "createUser", "commentsUser", "proj", "year"])
    return data_issue


def load_pr_data_proj(proj: str):
    filename = proj.replace("/", "__")
    data_pr = pd.read_csv(f"{global_settings.NEW_DATA_DIR}pr/{filename}.csv",
                          usecols=["startTime", "createUser", "commentsUser", "proj", "year"])
    return data_pr


def load_commit_data_proj(proj: str):
    """
    The path of the folder that contains all the csv files for commits shall be given by some global variable.

    The original data file looks like:

    ,committer_id,project_id,created_at,proj_name
    0,2723,2968,2012-07-31 18:45:14,sinatra_sinatra
    1,2723,2968,2012-07-31 07:04:42,sinatra_sinatra
    2,5274,2968,2012-07-31 18:43:18,sinatra_sinatra

    According to function `__get_all_commits()` in source file `take_snapshot.py`, only two columns, `created_at` and
    `proj_name`, are used. Because we provide the `proj` as a parameter and expect this function to return data only
    belongs to the project specified, we only need the `created_at` information.

    Parameters
    ----------
    proj

    `proj` is the name of the project whose data we want to load.

    Returns
    -------

    Pandas data frame of commits loaded for project `proj`.
    The data should contain columns `created_at` and `proj_name`.

    """

    """
    WARNING:    I have notices some differences between the csv file used in FSE and the new ones:

                1) The first column is always 0, which should be a increasing index starting from 0.
                    * fixed

                2) Header line not provided in commit related csv files. Should look like:
                   ',committer_id,project_id,created_at,proj_name', change according to the actual data columns.
                   * fixed

                3) So far as I know, `committer_id` , and `project_id` are not used in the code. I think we can replace 
                   `committer_id` with `committer_name`, and remove `project_id` since `project_name` is already given.
                   * fixed
                   
                4) Project name, use '/' or '_'? My suggestion is to use '_'.
                   * keep /, fix file name here 
                
                5) The date is formatted as '2012-07-30 18:05:51' in the original file, but as '2021-11-03T13:36:35Z' in
                   the new files. Suggest we use the new format, and modify the code in function `__get_all_commits()` 
                   in source file `take_snapshot.py`.
                   
                   * ToDo: modify the code mentioned above
                   * Done, we leave two options in the code, which can be controlled by global_settings.USE_NEW_DATA
                   
                6) In source file `take_snapshot.py`, functions `__get_commit_count_one_window()` and  `__get_one_window()`
                   extract data windows with `lambda x: start_time <= x < end_time`. However, the `start_time`, `x`, and
                   `end_time` are strings instead of datatime objects. I want to make sure that the lambda expression
                   can correctly selects a time window between `start_time`, and `end_time`.
                   
                   * ToDo: perform some tests to make sure the current implementation is correct.
                   * Tested, all pass. But I still prefer to keep the new one
    """

    filename = proj.replace("/", "__")
    data = pd.read_csv(f"{global_settings.NEW_DATA_DIR}commit/{filename}.csv")
    print(data)
    return data


def load_proj_data_time():
    data_time = pd.read_csv(f"proj_list/704/repoTimeCheck.csv",
                            usecols=["owner", "name", "repoTime", "issueTime", "PRTime", "commitTime"])
    print(data_time)
    return data_time


if __name__ == "__main__":
    proj = "Compass/compass"
    df = load_issue_pr_data_proj(proj)
    filename = proj.replace("/", "__")
    df.to_csv(f"../test_load_data_{filename}.csv")
