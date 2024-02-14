from statistics import mean
import pandas as pd
import numpy as np

from utilsPY import openreadtxt

def extract_first(data):
    try:
        split=[float(x[0].split(",")[1]) for x in data]
        shrink=[float(x[0].split(",")[2]) for x in data]
        merge=[float(x[0].split(",")[3]) for x in data]
        expand=[float(x[0].split(",")[4]) for x in data]
        return mean(split),mean(shrink),mean(merge),mean(expand),\
               np.var(split),np.var(shrink),np.var(merge),np.var(expand)
    except FileNotFoundError:
        return "none"
def max_index(list):
    index=0
    max=0
    for i in range(1,len(list)):
        if float(list[i]) >max:
            index=i
            max=float(list[i])
    return index

def extract_second(data):
    try:
        count=[0,0,0,0]
        area=[0,0,0,0]
        for dd in data:
            d=dd[0].split(",")[1:]
            d=[float(x) for x in d]
            count[max_index(d)]+=1
            mean_now=mean(d)
            for i in range(4):
                area[i]=area[i]+(d[i]-mean_now)
        count_result=[x/len(data) for x in count]
        return count_result,area

    except FileNotFoundError:
        return "none"
def feature_exa():
    # 1.read repoList
    re = openreadtxt("result/label.csv")[1:]
    features = []
    for rr in re:
        r = rr[0].split(",")
        data=openreadtxt("project_result/" + r[0].replace("/","__") + "_indexes.csv")[1:]
        bb = extract_first(data)
        if bb == "none":
            continue
        split_mean, shrink_mean, merge_mean, expand_mean, split_var, shrink_var, merge_var, expand_var \
            = bb
        aa = extract_second(data)
        if aa == "none":
            continue
        split_count, shrink_count, merge_count, expand_count = aa[0]
        split_area, shrink_area, merge_area, expand_area = aa[1]
        features.append([split_mean, shrink_mean, merge_mean, expand_mean,
                         split_var, shrink_var, merge_var, expand_var,
                         split_count, shrink_count, merge_count, expand_count,
                         split_area, shrink_area, merge_area, expand_area])
    dd = pd.DataFrame(features, columns=["split_mean", "shrink_mean", "merge_mean", "expand_mean",
                                         "split_var", "shrink_var", "merge_var", "expand_var",
                                         "split_count", "shrink_count", "merge_count", "expand_count",
                                         "split_area", "shrink_area", "merge_area", "expand_area"])
    dd.to_csv("result/features.csv", index=False)
if __name__=="__main__":
    feature_exa()