import datetime
def openreadtxt(file_name):
    data = []
    file = open(file_name, 'r')  # 打开文件
    file_data = file.readlines()  # 读取所有行
    for row in file_data:
        tmp_list = row.split(' ')  # 按‘，’切分每行的数据
        tmp_list[-1] = tmp_list[-1].replace('\n','') #去掉换行
        data.append(tmp_list)  # 将每行数据插入data中
    return data

def convertTime(strTime):
    strTime=strTime[:len(strTime)-1].replace("T"," ")
    # print(strTime)
    d1 = datetime.datetime.strptime(strTime, '%Y-%m-%d %H:%M:%S')
    return d1