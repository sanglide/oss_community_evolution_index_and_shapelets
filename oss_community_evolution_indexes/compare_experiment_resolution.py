import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

# Function to calculate RMSE between two series
def calculate_rmse(series_a, series_b):
    return sqrt(mean_squared_error(series_a, series_b))

def compare(A,B):
    re=[]
    # Calculate RMSE for each column and store results
    rmse_results = {}
    for column in A.columns:
        rmse_results[column] = calculate_rmse(A[column], B[column])

    # Display RMSE for each column
    for column, rmse in rmse_results.items():
        print(f"RMSE for column {column}: {rmse}")
        re.append(rmse)

    # Optionally, compare overall RMSE across all columns
    overall_rmse = sqrt((((A - B) ** 2).sum()).sum() / (len(A.columns) * len(A)))
    print(f"Overall RMSE across all columns: {overall_rmse}")
    re.append(overall_rmse)
    return re

if __name__=="__main__":
    result_root_path = ""
    result_path_list = []
    basic_path=""


    re=[]
    for c in result_path_list:
        print(f'--- {c} ---')
        A=pd.read_csv(f'{result_root_path}{basic_path}/index_productivity.csv')[['split','shrink','merge','expand']]
        B=pd.read_csv(f'{result_root_path}{c}/index_productivity.csv')[['split','shrink','merge','expand']]
        re.append(compare(A,B))
    df=pd.DataFrame(re)
    df.to_csv("rmse.csv")