import sys

if len(sys.argv[1]) == 0:
    filename='dump.txt'
else:
    filename=sys.argv[1]

f = open(filename)

lines = f.readlines()

candi_domain=[]

for line in lines:
    
    elements = line.split(' ')
    
    candi_num = []

    for i, element in enumerate(elements):
        if 'A?' == element:
            candi_num.append(i+1)
        elif 'AAAA?' == element: 
            candi_num.append(i+1)
        elif 'CNAME' == element:
            candi_num.append(i+1)

    for j in candi_num:
        candi_domain.append(elements[j])
    
# Erase duplicated domain
domain= []
for each in candi_domain:
    if each[-1] == ',':
        each = each[:-2]
        domain.append(each)

domain=list(set(domain))



filepath=filename+'_extracted_domain'
with open(filepath, 'w+') as lf:
    lf.write('\n'.join(domain))


