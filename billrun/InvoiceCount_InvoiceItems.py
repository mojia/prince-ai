import os


dir = '/Users/xinwang/Downloads'
file_1988 = 'invoice_map_count_invoice_item_1988.csv'
file_3586 = 'invoice_map_count_invoice_item_3586.csv'


def totalCountInvoiceItems(fileName):
    total = 0
    with open(os.path.join(dir, fileName), 'r') as f:
        for line in f:
            line = line.strip().replace('"', '')
            array = line.split(',')

            total += int(array[1])

    return total


print(totalCountInvoiceItems(file_1988))
print(totalCountInvoiceItems(file_3586))
