#
# namedtuples and NamedTuples
#

import datetime

# it is popular to represent data with dicts
stock_price = {'closing_price': 102.06, 'data': datetime.date(2014, 8, 29), 'symbol': 'AAPL'}

# but one major problem is accessing dicts by key is error-prone
# e.g. this typo doesn't raise an error and is wrong
stock_price['cosing_price'] = 103.06

# instead, we might use a namedtuple
# which is a tuple with named slots
from collections import namedtuple

StockPrice0 = namedtuple('StockPrice', ['symbol', 'date', 'closing_price'])
price = StockPrice0('MSFT', datetime.date(2018, 12, 14), 106.03)

assert price.symbol == 'MSFT'
assert price.closing_price == 106.03



# There is also a typed variant of namedtuple
from typing import NamedTuple
class StockPrice(NamedTuple):
    # typing
    symbol: str
    date: datetime.date
    closing_price: float

    def is_high_tech(self) -> bool:
        """It's a class, so we can add methods too"""
        return self.symbol in ['MSFT', 'GOOG', 'FB', 'AMZN', 'AAPL']

price1 = StockPrice('MSFT', datetime.date(2018, 12, 14), 106.03)

assert price1.symbol == 'MSFT'
assert price1.closing_price == 106.03
assert price1.is_high_tech()


#
# Cleaning and Munging
#

# before using ugly data, we must parse:    convert strings to floats/ints before using
#                                           check for missing values
#                                           check for outliers
#                                           check for bad data
from dateutil.parser import parse
from typing import List

def parse_row(row: List[str]) -> StockPrice:
    symbol, date, closing_price = row
    return StockPrice(symbol=symbol, date=parse(date).date(), closing_price=float(closing_price))

# test the parser
stock = parse_row(["MSFT", "2018-12-14", "106.03"])
assert stock.symbol == "MSFT"
assert stock.date == datetime.date(2018, 12, 14)
assert stock.closing_price == 106.03

# What if there is bad data?
# Maybe we would rather return a None than crash the program

from typing import Optional
import re

def try_parse_row(row: List[str]) -> Optional[StockPrice]:
    symbol, date_, closing_price_ = row

    # symbol should be all capital letters
    # ^ means beginning of the string
    # $ means end of the string
    # [A-Z] means one of the characters in the ASCII range from A to Z
    # + means one or more of previous
    if not re.match(r"^[A-Z]+$", symbol):
        return None
    
    try:
        date = parse(date_).date()
    except ValueError:
        return None
    
    try:
        closing_price = float(closing_price_)
    except ValueError:
        return None
    
    return StockPrice(symbol, date, closing_price)

# should return None for errors
assert try_parse_row(["MSFT0", "2018-12-14", "106.03"]) is None
assert try_parse_row(["MSFT", "2018-12--14", "106.03"]) is None
assert try_parse_row(["MSFT", "2018-12-14", "x"]) is None

# But should return the same as before if data is good
assert try_parse_row(["MSFT", "2018-12-14", "106.03"]) == stock

# lets test it on some bad data
# we will only return the good rows and ignore the bad rows
import csv

data: List[StockPrice] = []

with open('StockPrices.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        maybe_stock = try_parse_row(row)
        if maybe_stock is None:
            print(f"skipping invalid row: {row}")
        else:
            print(f"valid row: {row}")
            data.append(maybe_stock)

# Next step is to look for outliers. 
# For example, one of the dates in StockPrices.csv is 3014
# Need to catch these things

for sp in data:
    if sp.date.year > 2020:
        print(f'Found bad outlier in data[{data.index(sp)}]: year = {sp.date.year}')

data[2] = StockPrice('FB', datetime.date(2014, 6, 20), 64.5)
print(f'Fixed:  {data[2]}')


#
# Manipulating Data
#

# This section is more of a general approach
# Will give a few useful examples

# Get highest-ever closing price for each stock:
#   (1) Create a defaultdict to keep track of highest prices
#   (2) Iterate over data, updating the defaultdict

from collections import defaultdict

max_prices = defaultdict(lambda: float('-inf'))

for sp in data:
    closing_price, symbol = sp.closing_price, sp.symbol
    if closing_price > max_prices[symbol]:
        max_prices[symbol] = closing_price
    
for key, value in max_prices.items():
    print(f'Max price for {key}:\t{value}')

# Get largest and smallest one-day percent chanes in dataset:
#   (1) Order prices by date
#   (2) Use `zip` to get (previous, current)-pairs
#   (3) Turn the pairs into "percent change" rows

from typing import List, Dict
from collections import defaultdict

# Collect StockPrices by symbol
prices: Dict[str, List[StockPrice]] = defaultdict(list)
for sp in data:
    prices[sp.symbol].append(sp)

# Order StockPrices by date
# Since a `StockPrice = (symbol, date, price)` is a tuple, `sort` will sort by "dictionary ordering"
prices = {symbol: sorted(symbol_prices) for symbol, symbol_prices in prices.items()}

print('Sorted by symbol, then date:')
for s in prices.keys():
    print(f"{s}: ")
    for sp in prices[s]:
        print(f"\t{sp}")

# compute sequence of day over day changes
def pct_change(yesterday: StockPrice, today: StockPrice) -> float:
    return today.closing_price / yesterday.closing_price - 1

# new type
class DailyChange(NamedTuple):
    symbol: str
    date: datetime.date
    pct_change: float

# aux function

def day_over_day_changes(prices: List[StockPrice]) -> List[DailyChange]:
    """Assumes prices are for one stock and are in order"""
    return [DailyChange(symbol=today.symbol, date=today.date, pct_change=pct_change(yesterday, today)) for yesterday, today in zip(prices, prices[1:])]

# daily percent changes recorded in a list
all_changes = [change for symbol_prices in prices.values() for change in day_over_day_changes(symbol_prices)]

# key gives a rule for when max is evaluated on only one thing
max_change = max(all_changes, key=(lambda change: change.pct_change))
print(f"Max Daily Change: {max_change}")

min_change = min(all_changes, key=(lambda change: change.pct_change))
print(f"Min Daily Change: {min_change}")

