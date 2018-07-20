import os

dir = '/Users/xinwang/Downloads'
time1988 = 'total-time-1988.csv'
time3586 = 'total-time-3586.csv'


def sumTime(fileName):
    sum = 0
    with open(os.path.join(dir, fileName), 'r') as firstFile:
        for line in firstFile:
            if line != "":
                sum += float(line) / 1000.0
    return sum


print("time1988:" + str(sumTime(time1988)))
print("\ntime3586:" + str(sumTime(time3586)))
