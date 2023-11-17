import os


base_dir = '/srv/nfs/share/server_1/Cryptoless/MA/Level_1/result/'


folders = os.listdir(base_dir)

print(len(folders))

print(folders)
for folder in folders:
    working_dir = folder
    E_file_name = base_dir+'/'+working_dir+'/'+'E_output.csv'
    U_file_name = base_dir+'/'+working_dir+'/'+'U_output.csv'
    N_file_name = base_dir+'/'+working_dir+'/'+'N_output.csv'


#folders = [d for d in os.listdir(base_dir+'.') if os.path.isdir(d)]

