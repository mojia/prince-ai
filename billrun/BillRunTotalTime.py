import os


dir = '/Users/xinwang/Downloads'
file_1988_Name = 'billRun_total_time_1988.csv'
file_3586_Name = 'billRun_total_time_3586.csv'


def countTotalTime(fileName):
    total = 0
    with open(os.path.join(dir, fileName), 'r') as f:
        for line in f:
            total += int(line.replace('"', ''))

    return total


print(countTotalTime(file_1988_Name))
print(countTotalTime(file_3586_Name))
