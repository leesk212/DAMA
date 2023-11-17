import subprocess
import sys

filename = 'db.txt'

f = open(filename)

db = f.readlines()

cnt = 0

while(cnt<int(sys.argv[1])):
    print("@@"*60)
    print("CNT: "+str(cnt)+'/'+sys.argv[1])
    print("@@"*60)
    cnt=cnt+1
    filename = './log/log_'+str(cnt)+'.txt'

    cmd = 'sudo tcpdump -i tap0 udp -c 3  > ' + filename
    subprocess.call(cmd,shell=True)

    cmd = 'python extract.py ' + filename
    subprocess.call(cmd,shell=True)

    target = open(filename+'_extracted_domain')

    target_domains = target.readlines()  
    print(target_domains)

    for target_domain in target_domains:

        if target_domain[-1] !='\n':
            target_domain+='\n'

        if target_domain in db:
            print('Attack state')

        else:
            print('Normal state')




    
