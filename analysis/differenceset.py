import os


dir = '/Users/xinwang/Downloads'
afile = '165-invoice.txt'
bfile = '178-invoice.csv'


def readData(filename):
    data = []
    with open(os.path.join(dir, filename), 'r') as f:
        for line in f:
            data.append(line.strip())

    return data


alist = readData(afile)
blist = readData(bfile)
bset = set(blist)
print('alist size:' + str(len(alist)))
print('bset size:' + str(len(bset)))

common = []
for invoice in alist:
    if invoice in bset:
        # print('in ' + invoice)
        common.append(invoice)
        pass
    else:
        print(invoice)

print('common size ' + str(len(common)))
