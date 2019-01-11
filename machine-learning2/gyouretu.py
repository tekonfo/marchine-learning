import csv
import json
from IPython import embed
import datetime
import numpy as np




# 道路のID達
# seg_id = ['78404', '80377', '121148', '132595', '154613' , '178157']
# dir_str = ['EB', 'NB', 'EB', 'NB', 'EB', 'NB']
seg_id = ['78404', '80377', '121148', '154613' , '178157']
dir_str = ['EB', 'NB', 'EB', 'EB', 'NB']
# 目的地
target_id = '78404'
date_from = '10/12/2012'
date_to = '10/21/2012'
time_set = ["12:00-1:00 AM", "1:00-2:00AM", "2:00-3:00AM", "3:00-4:00AM", "4:00-5:00AM", "5:00-6:00AM", "6:00-7:00AM", "7:00-8:00AM", "8:00-9:00AM", "9:00-10:00AM", "10:00-11:00AM", "11:00-12:00PM", "12:00-1:00PM", "1:00-2:00PM", "2:00-3:00PM", "3:00-4:00PM", "4:00-5:00PM", "5:00-6:00PM", "6:00-7:00PM", "7:00-8:00PM", "8:00-9:00PM", "9:00-10:00PM", "10:00-11:00PM", "11:00-12:00AM"]


# hashのデータを整形して配列にする
def edit_hash(data, target_id):
  array_data = []
  hour_before_data = 0
  for day_hash in data:
    child_array = []
    child_array.append(hour_before_data)
    for box in data[day_hash]:
      if box == target_id:
        hour_before_data = data[day_hash][box]
      child_array.append(data[day_hash][box])
    array_data.append(child_array)
  return array_data

def make_id_hash(road_id):
  root_hash = {}
  for id in road_id:
    root_hash[id] = []
  return root_hash

def check_err(arr):
  count = 1
  length = len(arr)
  try:
    while count < length:
      if arr[count].size != arr[count-1].size:
        return True
      count+=1
    return False
  except:
    embed()



#dataハッシュに一レコードの全情報が入るはず
def insert_hash(line, data):
  for time in time_set:
    key = line["Date"] + " " + time
    try:
      if key not in data:
        data[key] = {}
      data[key][line["Segment ID"]] = int(line[time])
    except:
      import traceback
      traceback.print_exc()
  return data

# start
root_hash = make_id_hash(seg_id)
date_from = datetime.datetime.strptime(date_from, '%m/%d/%Y')
date_to = datetime.datetime.strptime(date_to, '%m/%d/%Y')
data = {}

# file読み込み
with open('test.csv') as f:
    for line in csv.DictReader(f):
        date = datetime.datetime.strptime(line["Date"], '%m/%d/%Y')
        if line["Segment ID"] in seg_id:
          tmp_index = seg_id.index(line["Segment ID"])
          direct = dir_str[tmp_index]
          if date_from <= date <= date_to  and direct == line["Direction"]:
            data = insert_hash(line, data)

array = edit_hash(data, target_id)
arr = np.array(array)
if check_err(arr):
  print("length not match")

A = arr
d = int(A.shape[0] * 0.8)
training_data = np.split(A, [d])[0]
test_data = np.split(A,[d])[1]

print(training_data)
result = training_data[:,0] #結果要素の取り出し
element = training_data[:,1:test_data.shape[1]]
result = result.reshape((len(result)),1) #resultを1次元配列から2次元配列へ
pinvA = np.linalg.pinv(element)
Object = np.dot(pinvA,result)


#以下テストデータをチェック
test_result = test_data[:,0]
test_element = test_data[:,1:test_data.shape[1]]
test_result = test_result.reshape(len(test_result),1)
print("a = \n"+str(test_result)) #1次元配列
test_predict = np.dot(test_element,Object)
print("b = \n"+str(test_predict)) #2次元配列
#print( "invA=\n" + str(pinvA) )

print(test_result.shape)

print(np.hstack((test_result,test_predict)))
i = 0
err_rate = []
while i <= test_predict.shape[0]-1:
    err = (np.absolute(test_predict[i][0] - test_result[i][0])/test_result[i][0]) * 100
    err_rate.append(err)
    i+=1
print(err_rate)

sum_result = 0
for result in err_rate:
  sum_result += result

print(sum_result/len(err_rate))

