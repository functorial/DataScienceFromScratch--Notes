#
# k-Nearest Neighbors
#

# This is one of the simplest machine learning algos out there

# Basically it classifies new data points based on a majority vote of their nearest neighbors

from typing import List
from collections import Counter

def raw_majority_vote(labels: List[str]) -> str:
    votes = Counter(labels)
    # `most_common(1)` will return [(winner, count)]
    winner, _ = votes.most_common(1)[0]
    return winner

# But what if we have a tie?
# In that event, we can reduce k until there is a unique winner
def majority_vote(labels: List[str]) -> str:
    """Assumes labels are ordered by distance, smallest -> largest"""
    votes = Counter(labels)
    winner, count = votes.most_common(1)[0]
    num_winners = len([x for x in votes.items() if x[1] == count])
    if num_winners == 1:
        return winner
    else:
        # k = k-1
        return majority_vote(labels[:-1])

# Now we can write a classifier

# use NamedTuples instead of dicts
from typing import NamedTuple
import numpy as np

class LabeledPoint(NamedTuple):
    point: np.ndarray
    label: str

def knn_classify(k: int, labeled_points: List[LabeledPoint], new_point: np.ndarray) -> str:
    # order `labeled_points` by increasing distance from `new_point`
    by_distance = sorted(labeled_points, key=lambda p: np.linalg.norm(new_point - p.point))
    # find labels for k closest points
    k_nearest_labels = [p.label for p in by_distance[:k]]
    # let the points vote
    return majority_vote(k_nearest_labels)



#
# The Iris Dataset
#

# This is a famous dataset
# Contains measurements for 150 flowers representing three species of iris.
# Features: petal length, petal width, sepal length, sepal width, species
import requests

data = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
# generic data file
with open('iris.dat', 'w') as f:
    f.write(data.text)

# The data is comma-separated with fields:
#   `sepal_length, sepal_width, petal_length, petal_width, class`
# First row:
#   `5.1, 3.5, 1.4, 0.2, Iris-setosa`

# Load data as LabeledPoints
from typing import Dict
import csv

def parse_iris_row(row: List[str]) -> LabeledPoint:
    # e.g. measurements = [5.1, 3.5, 1.4, 0.2]
    measurements = np.array([float(value) for value in row[:-1]])
    # e.g. label == 'setosa'
    label = row[-1].split("-")[-1]

    return LabeledPoint(measurements, label)

# Parse data into a List[LabeledPoint]
with open('iris.dat') as f:
    reader = csv.reader(f)
    iris_data = []
    for r in reader:
        try: # To catch any empty rows
            pir = parse_iris_row(r)
            iris_data.append(pir)
        except IndexError:
            pass

# Group the points by species/label so we can plot them
# remember defaultdict is a dict that creates entries on invalid lookups
from collections import defaultdict
points_by_species: Dict[str, List[np.ndarray]] = defaultdict(list)
for iris in iris_data:
    points_by_species[iris.label].append(iris.point)


# Plot the data by scatterplots for each of the 4 choose 2 = 6 pairs of measurements
from matplotlib import pyplot as plt
metrics = ['sepal length', 'sepal width', 'petal length', 'petal width']
pairs = [(i,j) for i in range(4) for j in range(i)]
# 3 markers for 3 classes
marks = ['+', '.', 'x']

fig, ax = plt.subplots(2, 3)

for row in range(2):
    for col in range(3):
        i, j = pairs[3 * row + col]
        ax[row][col].set_title(f"{metrics[i]} vs {metrics[j]}", fontsize=8)

        # plot the scatterplots for each species
        for mark, (species, points) in zip(marks, points_by_species.items()):
            xs = [point[i] for point in points] # get ith measurement
            ys = [point[j] for point in points] # get jth measurement
            ax[row][col].scatter(xs, ys, marker=mark, label=species) # label is for the legend

ax[-1][-1].legend(loc='lower right', prop={'size': 6}) # prop means font properties
plt.tight_layout()
plt.show()              # We can see that the data really is clustered by species
                        # So a k-Nearest Neighbors algorithm really does make sense

# Split the data into training, test
import random
from MachineLearning import split_data
random.seed(12)

pct = 0.40   # Feel free to mess around with this number!!!!!!!!!!!!!!
iris_train, iris_test = split_data(iris_data, pct)
assert len(iris_train) == 150 * pct
assert len(iris_test) == 150 * (1 - pct)

# The training set will be the neighbors 
# used for classifying the test set points

# We might create some machinery to choose k
# or we can go with our gut

from typing import Tuple

k_values = [x for x in range(1, 151)]
ys = []

for k in k_values:
    print(f"Working on k = {k} ...")
    # track how many times we see (predicted, actual)
    confusion_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
    num_correct = 0

    for iris in iris_test:
        predicted = knn_classify(k, iris_train, iris.point)     # feel free to change k 
        actual = iris.label

        if predicted == actual:
            num_correct += 1

        confusion_matrix[(predicted, actual)] += 1

    # Compute and print the accuracy of the model
    pct_correct = num_correct / len(iris_test)
    print(f"Percent Correct: {pct_correct * 100}%")
    print(f"Confusion Matrix: {confusion_matrix}")
    # Might want to check f1-scores in other situations....

    ys.append(pct_correct)

print(f"x length: {len(k_values)}")
print(f"y length: {len(ys)}")

plt.bar(k_values, ys)
plt.xlabel('Neighbors')
plt.ylabel('Accuracy')
plt.title(f'Optimal choice of neighbors? From {int(pct*100)}% training data.')
plt.show()

