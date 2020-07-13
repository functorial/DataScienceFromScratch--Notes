import csv

# working with a .txt file with a header
# organizes data into a dict using header as the dict structure
# this way is better, I think
# if file had no header, we could have provided our own
with open('ColonDelimitedStockPricesHeader.txt', newline='') as f:
    reader = csv.DictReader(f, delimiter=":")
    fieldnames = reader.fieldnames
    with open('CommaDelimitedStockPricesHeader.txt', 'w', newline='') as n:
        writer = csv.DictWriter(n, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()
        for row in reader:
            writer.writerow(row)

            
# working with a .txt file without a header
with open('CommaDelimitedStockPrices.txt', newline='') as f:
    tab_reader = csv.reader(f, delimiter=',')
    with open('ColonDelimitedStockPrices.txt', 'w', newline='') as n:
        csv_writer = csv.writer(n, delimiter=":")
        for row in tab_reader:
            print(row)
            date = row[0]
            symbol = row[1]
            closing_price = float(row[2])
            csv_writer.writerow([date, symbol, closing_price])