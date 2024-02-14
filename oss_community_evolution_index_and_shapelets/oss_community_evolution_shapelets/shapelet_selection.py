import classify_supervised_learning

if __name__=="__main__":
    time_of_execution="2023-03-27T22-22-43Z"
    for i in range(100,0,-5):
    # for i in range(5,0,-1):
        print("nn_shapelets : {0}".format(i))
        classify_supervised_learning.script_classification_ml_multi_sizes(time_of_execution,i)