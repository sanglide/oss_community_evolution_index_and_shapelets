import copy
import csv
import statsmodels.stats.multicomp as sm
import numpy as np
import pandas as pd
from pandas import Series
from scipy import stats
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

import global_settings


def is_normal_distribution_single_series(d):
    """

    Parameters
    ----------
    d

    Returns
    -------
    boolean value(True/False)
    If data matches the normal distribution(p>0.05), this function returns "True"
    """
    data = Series(d)
    average = data.mean()  # average value
    std = data.std()  # standard deviation
    k = stats.kstest(data, 'norm', (average, std))
    # KS inspection returns statistic value( D value) and P value
    if k.pvalue > 0.05:
        return True
    else:
        return False


def check_normal_distribution(index_file_path):
    csvFile = open(index_file_path, "r")
    reader = csv.reader(csvFile)
    title = []
    column_dict = {}
    for line in reader:
        if reader.line_num == 1:
            title = line
            column_dict = {title.index(x): [] for x in title}
            continue
        for j in range(0, len(line)):
            column_dict[j].append(float(line[j]))
    csvFile.close()
    for i in range(0, len(title)):
        if not is_normal_distribution_single_series(column_dict[i]):
            return False
    return True

def index_correlation_analysis(index_file_path, analysis_method_type):
    """

    Parameters
    ----------
    index_file_path

    The file path of a csv file that contains the values of the indexes, which is formatted as  follows:

    split,shrink,merge,expand
    0.0,0.0,0.0,2.4713055879414756
    0.0,0.0,0.0,0.11028927321055719
    13.146408959447172,2.0826733483226713e-15,0.0,0.035658124899604314
    0.0,0.0,0.0,4.701370952820387
    0.0,0.0,0.0,0.0
    45.369652324084946,0.0,42.9608597566134,1.8787119938195476
    43.30755426062091,2.707773865342931e-15,66.62245258232036,0.0
    59.802202838456544,0.0,56.27376051019882,0.5807123976090129
    ...

    The first line is the title line. Each line following the title line contains the values of the four indexes,
    separated by the comma. The lines are aligned by time, i.e., each column is a sequence of the index values
    indicated by the title line. Refer to the sample csv file.




    Returns
    -------

    A list of dict that contains the results of correlation tests for each pair of the indexes, i.e., 12 combinations,
    which looks like the following

    [{"pair": "split_shrink", "correlation": 0.1, "p": 0.001},  {"pair": "split_merge", "correlation": 0.2, "p": 0.1}, ...]

    where
    "pair" is the name of the pair
    "correlation" is the pearson product-moment correlation coefficient (PPMCC), i.e., the r value
    "p" is the p value, refer to Sec. 4.3 of paper [1]

    [1] U. Raja, and M. J. Tretter, 《Defining and Evaluating a Measure of Open Source Project Survivability》,
      IEEE Transactions on Software Engineering, 卷 38, 期 1, 页 163–174, 1月 2012, doi: 10.1109/TSE.2011.39.


    questions need to be answered:
    Q1: What is the difference between PPMCC and Pearson's Correlation Coefficent? Is PPMCC just a cool name?
    A1:

    """

    # load the file, whose path is given by `index_file_path`
    csvFile = open(index_file_path, "r")
    reader = csv.reader(csvFile)
    title, result = [], []
    column_dict = {}
    for line in reader:
        if reader.line_num == 1:
            title = line[1:]
            column_dict = {title.index(x): [] for x in title}
            continue
        for j in range(len(line[1:])):
            column_dict[j].append(float(line[1:][j]))
    csvFile.close()
    # "title" is an array including name of each column, "column_dict" contains data of each column
    # boolean array, judge whether each column is a normal distribution
    # for each pair of the indexes
    reject=[]
    for i in range(len(title)):
        for j in range(i + 1, len(title)):
            # print("column index 1:{0} column index 2:{1}".format(i,j))
            if analysis_method_type == 'pearson':  # all columns fit the normal distribution.
                r, p = pearsonr(column_dict[i], column_dict[j])
                # print("pearsin: {0} {1}".format(r,p))
            elif analysis_method_type == 'spearman':
                sp = stats.spearmanr(column_dict[i], column_dict[j])
                r, p = sp.correlation, sp.pvalue
                # print("spearman: {0} {1}".format(r, p))
            elif analysis_method_type=="bonferroni":
                sp = stats.spearmanr(column_dict[i], column_dict[j])
                r, p = sp.correlation, sp.pvalue
                reject_1, pvals_corrected_1, alphacSidak1, alphacBonf1 = multipletests(p, 0.001,method='bonferroni')
                reject_2, pvals_corrected_2, alphacSidak2, alphacBonf2 = multipletests(p, 0.01,method='bonferroni')
                reject_3, pvals_corrected_3, alphacSidak3, alphacBonf3 = multipletests(p, 0.05,method='bonferroni')
                reject.append([reject_1,reject_2,reject_3])


            # elif analysis_method_type=="FDR":
            #     sp = stats.spearmanr(column_dict[i], column_dict[j])
            #     r, p = sp.correlation, sp.pvalue
            #     p_adjusted = multipletests(p, method='fdr_bh')[1]
            #     p = p_adjusted

            result.append({"pair": title[i] + '_' + title[j], "correlation": r, "p": p})
    # calculate the r value and the p value, append the results in the list
    # call existing libraries, do not implement it yourself

    # return the list of dicts as required

    return result,reject


def index_commit_correlation_analysis(index_commit_file_path):
    """

    Parameters
    ----------
    index_commit_file_path

    The file path of a csv file that contains the values of the indexes, which is formatted as  follows
    (the commit_count values in this example are pseudo values):

    split,shrink,merge,expand,commit_count
    0.0,0.0,0.0,2.4713055879414756,100
    0.0,0.0,0.0,0.11028927321055719,110
    13.146408959447172,2.0826733483226713e-15,0.0,0.035658124899604314,10
    0.0,0.0,0.0,4.701370952820387,20
    0.0,0.0,0.0,0.0,33
    45.369652324084946,0.0,42.9608597566134,1.8787119938195476,50
    43.30755426062091,2.707773865342931e-15,66.62245258232036,0.0,12
    59.802202838456544,0.0,56.27376051019882,0.5807123976090129,123

    Returns
    -------

    Statistical test results that reveal the relations between the independent variables and the dependent variable:

    Dependent variable: `commit_count` itself, or other values that can be derived from `commit_count`, e.g., the change
                        (difference) in `commit_count` over time steps, the cumulated value of `commit_count`, and etc

    Independent variables: `split`, `shrink`, `merge`, and `expand`, or other values that can be derived from these raw
                           index values. We once found that the raw `split`, `shrink`, `merge`, and `expand` values have
                           a strong correlation with the change of `commit_count` over time steps, this is potentially
                           because these indexes evaluate the 'change' or 'evolution' of communities, which directly
                           relate to changes in commit counts.

    You may use tools like ANOVA to perform the analysis, besides the main effects, we also need to find the interaction
    between the independent variables. You can also refer to the paper and try the least-squares dummy variable (LSDV)
    used by Li Ying in the original manuscript. I am less certain on what variables and which tool to use to perform
    the analysis, you can consult Zhiwen Zheng for his way of performing correlation analysis about project productivity.
    Maybe we also need to apply some preprocessing steps, you can ask him.

    To save your effort and to make it easier to try different approaches, I suggest:
    1) generate and store a series of variables given the raw data, as suggested above;
    2) make the code configurable, so we can try different combinations of variables, preprocessing steps, and testing tools;
    3) provide citations of OSS related research papers to support the way you used in analyzing our data.

    """
    pass


def __get_discriminant_validity_latex_table_str(correlation_list):
    # the key point is whether the indexes CAN show some independence
    def corr2tag(number):
        if number == 1 or number == -1:
            return "\\colorbox{red}{P}"  # "Perfect correlation"
        elif number >= 0.7 or number <= -0.7:
            return "\\colorbox{pink}{S}"  # "Strong correlation"
        elif number >= 0.4 or number <= -0.4:
            return "\\colorbox{yellow}{M}"  # "Moderate correlation"
        elif number >= 0.1 or number <= -0.1:
            return "\\colorbox{lime}{W}"  # "Weak correlation"
        else:
            return "\\colorbox{white}{N}"  # "No correlation"

    def p2stars(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return ""
    def p2stars_B(value):
        if value[0]==True:
            return "***"
        elif value[1]==True:
            return "**"
        elif value[2]==True:
            return "*"
        else:
            return ""


    index_names = ['split', 'shrink', 'merge', 'expand','developers_count','developers_change_count']

    table_head_str = f"%global_settings.EVOLUTION_PATTERN_THRESHOLD = {global_settings.EVOLUTION_PATTERN_THRESHOLD}\n" \
                     "\\begin{table*}[!t]\n" \
                     "\\renewcommand{\\arraystretch}{1.5}\n" \
                     "\\addtolength{\\tabcolsep}{-2pt} %tight up the left and right\n" \
                     "\\newcommand{\\tabincell}[2]{\\begin{tabular}{@{}#1@{}}#2\end{tabular}}\n" \
                     "\\centering\n" \
                     "\\footnotesize\n" \
                     "\\caption{Pairwise correlation of the aggregated indexes (Spearman's Correlation Coefficient). Project No. follow Table \\ref{tbl:data_set}. Notations follow Table \\ref{tbl:corr_rules}.  The significance levels are shown without the $p$ values to make the table concise. Better viewed in color.}\\label{tbl:correlation}\n" \
                     "\\begin{tabular}{c|"
    for i in range(len(index_names)):
        for j in range(i + 1, len(index_names)):
            if index_names[j] != 'developers_count' and index_names[i] != 'developers_count':
                table_head_str = table_head_str + "l"
    table_head_str = table_head_str + "}\n\hline\n"

    table_tail_str = "\\multicolumn{7}{l}{Significance level: ***p \\textless 0.001, **p \\textless 0.01, *p \\textless 0.05}\\end{tabular}\\end{table*}\n"

    latex_table = ""
    latex_table = latex_table + table_head_str
    # the title line
    latex_table = latex_table + "\\textbf{Proj. No.}"
    for i in range(len(index_names)):
        for j in range(i + 1, len(index_names)):
            if index_names[j]!='developers_count' and index_names[i]!='developers_count':
                latex_table = latex_table + f"&{index_names[i]}-{index_names[j]}"
    latex_table = latex_table + "\\\\ \\hline\n"

    project_no = 1

    for corr in correlation_list:
        # latex_table = latex_table + str(corr['project']).replace('_', '/')
        if project_no <= len(global_settings.proj_list):
            latex_table = latex_table + str(project_no)
        else:
            latex_table = latex_table + "\\textbf{Overall}"
        project_no = project_no + 1
        idx = 0
        for i in range(len(index_names)):
            for j in range(i + 1, len(index_names)):
                # latex_table = latex_table + "\t&\t" + "\\tabincell{l}{r:" + "{:.2f}".format(
                #     corr['results'][idx]['correlation']) + f" ({corr2tag(corr['results'][idx]['correlation'])})\\\\p:" + \
                #               "{:.2f}".format(corr['results'][idx]['p']) + p2stars(corr['results'][idx]['p'])+"}"

                # latex_table = latex_table + "\t&\t" + "r: " + "{:.2f}".format(
                #     corr['results'][idx]['correlation']) + f" ({corr2tag(corr['results'][idx]['correlation'])}), p: " + \
                #               "{:.2f}".format(corr['results'][idx]['p']) + p2stars(corr['results'][idx]['p'])
                if project_no <= len(global_settings.proj_list):
                    if index_names[j] != 'developers_count' and index_names[i] != 'developers_count':
                        if global_settings.CORRELATION_METHOD=="bonferroni":
                            latex_table = latex_table + "\t&\t" + \
                                          ("\\textcolor{white}{~}" if corr['results'][idx]['correlation'] >= 0 else " ") + \
                                          "{:.2f}".format(corr['results'][idx]['correlation']) + \
                                          f" ({corr2tag(corr['results'][idx]['correlation'])})" + \
                                          p2stars_B(corr['reject'][idx])
                        else:

                            latex_table = latex_table + "\t&\t" + \
                                          ("\\textcolor{white}{~}" if corr['results'][idx]['correlation'] >= 0 else " ") + \
                                          "{:.2f}".format(corr['results'][idx]['correlation']) + \
                                          f" ({corr2tag(corr['results'][idx]['correlation'])})" + \
                                          p2stars(corr['results'][idx]['p'])

                    print(f"{index_names[i]}_{index_names[j]}")
                    # print(corr['results'][idx]['pair'])
                    assert (f"{index_names[i]}_{index_names[j]}" == corr['results'][idx]['pair'])
                idx = idx + 1

        latex_table = latex_table + "\\\\ \n"
    latex_table = latex_table + "\\hline \n"
    latex_table = latex_table + table_tail_str
    return latex_table


def execute_index_independency_check(result_folder_path):
    fo = open(result_folder_path + "discrimant_validity.txt", "a+")

    correlation_list = []
    proj_list = copy.deepcopy(global_settings.proj_list)
    proj_list.append('total')


    analysis_method_type = global_settings.CORRELATION_METHOD
    print(f"\n\n {analysis_method_type}'s Correlation \n\n")
    fo.write("\n\n {analysis_method_type}'s Correlation \n\n")

    # peason's or spearman's

    for proj in proj_list:
        proj_name = proj.replace("/", "__")
        r=index_correlation_analysis(
            result_folder_path + "community_evolution/" + proj_name + "_indexes.csv", analysis_method_type)
        temp_correlation_dict = {'project': proj, 'results': r[0],'reject':r[1]}
        correlation_list.append(temp_correlation_dict)
        fo.write(str(temp_correlation_dict) + "\n")
        # print(str(temp_correlation_dict))
    latex_table_str = __get_discriminant_validity_latex_table_str(correlation_list)
    print(latex_table_str)
    fo.write("\n" + latex_table_str + "\n")

    fo.flush()
    fo.close()
    return correlation_list


def prepare_data_productivity(projects_aggregated_split, projects_aggregated_shrink, projects_aggregated_merge,
                              projects_aggregated_expand, projects_commit_count, projects_snapshot_start_time_list,
                              projects_snapshot_issue_count_list, projects_snapshot_pr_count_list,
                              projects_snapshot_issue_pr_count_list, projects_snapshot_member_count_list,projects_member_change_count_list,
                              result_folder_path):
    fo = open(result_folder_path + "index_productivity.csv", 'w')
    fo.write(
        'project_name,time,split,shrink,merge,expand,commit_count,commit_count_diff,project_age,issue_count,pr_count,issue_pr_count,member_count,member_change_count\n')
    proj_idx = 0
    for proj in global_settings.proj_list:
        assert len(projects_snapshot_start_time_list[proj_idx]) - 1 == len(projects_aggregated_split[proj_idx])
        assert len(projects_commit_count[proj_idx]) - 1 == len(projects_aggregated_split[proj_idx])
        for idx in range(len(projects_aggregated_split[proj_idx])):
            fo.write(
                f'{proj},{projects_snapshot_start_time_list[proj_idx][idx]},{projects_aggregated_split[proj_idx][idx]},{projects_aggregated_shrink[proj_idx][idx]},'
                f'{projects_aggregated_merge[proj_idx][idx]},{projects_aggregated_expand[proj_idx][idx]},'
                f'{projects_commit_count[proj_idx][idx]},'
                f'{projects_commit_count[proj_idx][idx + 1] - projects_commit_count[proj_idx][idx]},{idx},'
                f'{projects_snapshot_issue_count_list[proj_idx][idx]},{projects_snapshot_pr_count_list[proj_idx][idx]},'
                f'{projects_snapshot_issue_pr_count_list[proj_idx][idx]},{projects_snapshot_member_count_list[proj_idx][idx]},{projects_member_change_count_list[proj_idx][idx]}\n')
            # f'{projects_commit_count[proj_idx][idx] - projects_commit_count[proj_idx][idx - 1] if idx > 1 else 0},{idx}\n')
        proj_idx = proj_idx + 1

    fo.flush()
    fo.close()


def prepare_data_active_inactive_pred_single_proj(proj_aggregated_split, proj_aggregated_shrink,
                                                  proj_aggregated_merge, proj_aggregated_expand, proj_name_by_slash,
                                                  result_folder_path):
    # determine the status of the project as active / inactive with the last commit time
    # extract features (with different window size ahead of the last commit)
    # return data that meets the format of R script for logistic regression analysis
    pass
