from sklearn.preprocessing import RobustScaler, StandardScaler
import pandas as pd
import tensorflow as tf
import sys
import subprocess
import signal
import time
import os
import numpy as np

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
#            print(pre_each)
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

            #if float(pre_each[0]) not in time:
            #    time.append(float(pre_each[0]))


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


f = open('level_1_result.txt','w')

# 0. Load 


base_dir = './model/'
##E
E_model=tf.keras.models.load_model(base_dir+'E_model.hd')
##U
U_model=tf.keras.models.load_model(base_dir+'U_model.hd')
##N
N_model=tf.keras.models.load_model(base_dir+'N_model.hd')



# 1. read Data 1.5, Preprocessing Data

base_dir = '/srv/nfs/share/server_1/Cryptoless/MA/Level_1/result/'

#folders = [d for d in os.listdir(base_dir+'.') if os.path.isdir(d)]
folders = os.listdir(base_dir)

print(len(folders))

for folder in folders:
    if folder == 'Readme.md':
        continue
    working_dir = folder
    E_file_name = base_dir+'/'+working_dir+'/'+'E_output.csv'
    U_file_name = base_dir+'/'+working_dir+'/'+'U_output.csv'
    N_file_name = base_dir+'/'+working_dir+'/'+'N_output.csv'


    scaler = StandardScaler()



## E
    E_raw = U_preprocessed(E_file_name)

    #print(E_raw.describe())

    E_raw = E_raw.head(200)
    #E_raw = E_raw.drop(['Unnamed: 0'],axis=1)
    E_raw = scaler.fit_transform(E_raw)
    try: 
        E_raw = E_raw.reshape(20,10,7)
    except:
        E_raw = pd.DataFrame(E_raw)
        E_raw = E_raw.head(100)
        E_raw = scaler.fit_transform(E_raw)
        E_raw = E_raw.reshape(10,10,7)

    #print('S'+str(type(E_raw)))

    E_result = E_model.predict(E_raw)
    Final_E = 0
    if E_result[0][0] > 0.5:
        Final_E=1



## U
    U_raw = pd.read_csv(U_file_name)
    
    U_raw = U_raw.fillna(0)
    first_column_name = U_raw.columns[0]

# 첫 번째 열 삭제
    U_raw = U_raw.drop(columns=first_column_name)

    U_raw = pd.concat([U_raw, U_raw], ignore_index=True)

    U_raw = scaler.fit_transform(U_raw)

    #print(U_raw.info())
        
    #U_raw = U_raw.to_numpy()    

    try:
        U_raw = U_raw.reshape(1,10,10)
        U_result = U_model.predict(U_raw)
        Final_U = 0
        if U_result[0][0] > 0.5:
            Final_U=1
    except:
        Final_U=0




## N
    try: 
    
        N_raw = pd.read_csv(N_file_name)
        N_raw = N_raw.fillna(0)
        N_raw = scaler.fit_transform(N_raw)
        N_raw = N_raw.reshape(1,10,11)


        N_result = N_model.predict(N_raw)
        Final_N = 0
        if N_result[0][0] > 0.5:
            Final_N=1
    except:
        Final_N = 0



# 2. Classifiy Current state by classifier

## E
#    E_result = E_model.predict(E_raw)
#    Final_E = 0
#    if E_result[0][0] > 0.5:
#        Final_E=1


## U
#    U_result = U_model.predict(U_raw)
#    Final_U = 0
#    if U_result[0][0] > 0.5:
#        Final_U=1

## N
#    N_result = N_model.predict(N_raw)
#    Final_N = 0
#    if N_result[0][0] > 0.5:
#        Final_N=1



    # 3. Majority vote 
    if Final_E+Final_U+Final_N+1 > 2:
        print(1)
        f.write(str(1)+'/'+str(Final_E)+'/'+str(Final_U)+'/'+str(Final_N)+'/'+' 1\n')
    else:
        print(0)
        f.write(str(1)+'/'+str(Final_E)+'/'+str(Final_U)+'/'+str(Final_N)+'/'+' 0\n')



