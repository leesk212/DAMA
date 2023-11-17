from sklearn.preprocessing import RobustScaler, StandardScaler
import pandas as pd
import tensorflow as tf
import sys
import subprocess
import signal
import time


def U_preprocessed(filename):

    f = open(filename,'r')
    lines = f.readlines()

    uops_0 = []
    uops_1 = []
    uops_2_3 = []
    uops_4_9 = []
    uops_5 = []
    uops_6 = []
    uops_7_8 = []
    feature = [uops_0,uops_1,uops_2_3,uops_4_9,uops_5,uops_6,uops_7_8]

    for i,line in enumerate(lines):
        if i > 1:
            pre_each = line.strip('').split(',')

            if str(pre_each[3].split('_')[-1]) == '0':
                try:
                    uops_0.append(int(pre_each[1]))
                except:
                    uops_0.append(0)
            if str(pre_each[3].split('_')[-1]) == '1':
                try:
                    uops_1.append(int(pre_each[1]))
                except:
                    uops_1.append(0)

            if str(pre_each[3].split('_')[-1]) == '3':
                try:
                    uops_2_3.append(int(pre_each[1]))
                except:
                    uops_2_3.append(0)
            if str(pre_each[3].split('_')[-1]) == '9':
                try:
                    uops_4_9.append(int(pre_each[1]))
                except:
                    uops_4_9.append(0)
            if str(pre_each[3].split('_')[-1]) == '5':
                try:
                    uops_5.append(int(pre_each[1]))
                except:
                    uops_5.append(0)
            if str(pre_each[3].split('_')[-1]) == '6':
                try:
                    uops_6.append(int(pre_each[1]))
                except:
                    uops_6.append(0)
            if str(pre_each[3].split('_')[-1]) == '8':
                try:
                    uops_7_8.append(int(pre_each[1]))
                except:
                    uops_7_8.append(0)

            if float(pre_each[0]) not in time:
                time.append(float(pre_each[0]))


    df = pd.DataFrame({
                   "0":uops_0,
                   "1":uops_1,
                   "2":uops_2_3,
                   "3":uops_4_9,
                   "4":uops_5,
                   "5":uops_6,
                   "6":uops_0})
        
    return df





#debug info off
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(3)



# 0. Load 


base_dir = './models/level1/'
##E
E_model=tf.keras.models.load_model(base_dir+'E_model.hd')
##U
U_model=tf.keras.models.load_model(base_dir+'U_model.hd')
##N
N_model=tf.keras.models.load_model(base_dir+'N_model.hd')



# 1. read Data 1.5, Preprocessing Data

base_dir = '/srv/nfs/share/'
working_dir = sys.argv[1]
E_file_name = base_dir+working_dir+'E_output.csv'
U_file_name = base_dir+working_dir+'U_output.csv'
N_file_name = base_dir+working_dir+'N_output.csv'


scaler = StandardScaler()



## E
E_raw = U_preprocessed(E_file_name)
E_raw = E_raw.head(7000)
E_raw = E_raw.drop(['Unnamed: 0'],axis=1)
E_raw = scaler.fit_transform(E_raw)
E_raw = E_raw.reshape(700,10,7)

## U
U_raw = pd.read_csv(U_file_name)

## N
N_raw = pd.read_csv(N_file_name)
N_raw = N_raw.fillna(0)
N_raw = scaler.fit_transform(N_raw)
N_raw = N_raw.reshape(1,10,11)





# 2. Classifiy Current state by classifier

## E
E_result = E_model.predict(E_raw)
Final_E = 0
if E_result[0][0] > 0.5:
    Final_E=1


## U
U_result = U_model.predict(U_raw)
Final_U = 0
if U_result[0][0] > 0.5:
    Final_U=1

## N
N_result = N_model.predict(N_raw)
Final_N = 0
if N_result[0][0] > 0.5:
    Final_N=1



# 3. Majority vote 
f = open('level_1_result.txt','a')
if Final_E+Fianl_U+Final_N+1 > 2:
    f.write('1\n')



