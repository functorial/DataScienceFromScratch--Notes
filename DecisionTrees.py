#
# Decision Trees
#

# A decision tree is a model that forms a tree-like structure
#   like the game 'Twenty Questions'

# Pros:
#   (*) Can handle data of various types, e.g. numeric and categorical
#   (*) Easy to understand
#   (*) Easy to interpret 
#
# Cons:
#   (*) Computationally very hard to find an optimal descision tree
#           Get around this by building one that is 'good enough'
#   (*) Very easy to overfit to data
#           e.g. a 'guess the animal' game with a 3 layer deep tree

# Classification Trees:
#   Produce categorical outputs
#
# Regression Trees:
#   Produce numerical outputs

# Two things to decide:
#   (1) Which questions to ask?
#   (2) In what order?

# Would like to choose questions whose answers 'give a lot of information'
#   Regard a 'question' as a partition of the data into classes (based on their answers tot he question)
#   We will quantify the notion of 'amount of info given by a partition' with /entropy/
#
# Entropy:
#   Let S be a set of data points, 
#   each labeled by a category from C_0, ..., C_{n-1}, 
#   and let p_i be the proportion of points in S labeled by C_i
#
#   Then, define the /entropy/ of S to be the quantity
#
#       H(S) = - (p_0 * log_2(p_0) + ... + p_n * log_2(p_{n-1}))
#            = - log_2(product( p_i ^ p_i for i in range(0, n) if p_i != 0 ))

# Note:
#   Use the convention 0 * log_2(0) = 0
#   The quantities -p_i * log_2(p_i) are between 0 and ~0.53, 
#       since p_i is between 0 and 1

# Graph of p log_2(p)
import matplotlib.pyplot as plt
import numpy as np

xs = [0] + [i/100 for i in range(1, 101)]
ys = [0] + [-x * np.log2(x) for x in xs if x != 0]
plt.plot(xs, ys)
plt.title('f(p) = -p * log_2(p)')
#plt.show()


# Define Entropy function:
def f(p):
    return -p * np.log2(p)

vf = np.vectorize(f)

def entropy(class_probabilities: np.ndarray) -> float:
    return np.sum(vf(class_probabilities))

# The entropy will be small when every p_i is close to 0 or 1
#   and will be larger when many of the p_i's are more evenly distributed 
assert entropy(np.array( [1.0] )) == 0.0
assert entropy(np.array( [0.5, 0.5] )) == 1.0

# After our data answers our question,
#   the data will consist of pairs (input, label)
#       we will need to compute the class probabilities ourself to compute the entropy
from collections import Counter

def data_entropy(labels: np.ndarray) -> float:
    total_count = np.size(labels)
    class_probabilities = np.array([count / total_count for count in Counter(labels).values()]) # we don't care about order
    return entropy(class_probabilities)

assert data_entropy(['a']) == 0
assert data_entropy([True, False]) == 1.0


# Entropy of a Partition:
#   Given a partition of a set S into subsets S_1, ..., S_n,
#   want to define the notion of /entropy of a partition/ so that:
#       (1) if all S_i have low entropy, 
#           then the partition should have low entropy
#       (2) if some S_i have high entropy (and are large), 
#           then the partition should have high entropy
#   Define as a weighted sum:
#       H = q_1 * H(S_1) + ... + q_n * H(S_n)

from typing import List

def partition_entropy(subsets: List[np.ndarray]) -> float:
    sizes = [len(subset) for subset in subsets]
    total_size = sum(sizes)
    proportions = np.array([size / total_size for size in sizes])
    entropies = np.array([data_entropy(subset) for subset in subsets])
    return np.dot(proportions, entropies)


#
# Creating a Decision Tree Example
#   Implementing the ID3 algorithm
#

from typing import NamedTuple, Optional

class Candidate(NamedTuple):
    level: str
    lang: str
    tweets: bool
    phd: bool
    did_well: Optional[bool] = None     # allow unlabeled data

inputs = [Candidate('Senior', 'Java', False, False, False),
          Candidate('Senior', 'Java', False, True, False),
          Candidate('Mid', 'Python', False, False, True),
          Candidate('Junior', 'Python', False, False, True),
          Candidate('Junior', 'R', True, False, True),
          Candidate('Junior', 'R', True, True, False),
          Candidate('Mid', 'R', True, True, True),
          Candidate('Senior', 'Python', False, False, False),
          Candidate('Senior', 'R', True, False, True),
          Candidate('Junior', 'Python', True, False, True),
          Candidate('Senior', 'Python', True, True, True),
          Candidate('Mid', 'Python', False, True, True),
          Candidate('Mid', 'Java', True, False, True),
          Candidate('Junior', 'Python', False, True, False)
          ]


# ID3 Algorithm:
#
#   Let S be a set of labeled data  (e.g. candidate list = List[Candidate, did_well])
#   Let A be a list of attributes   (e.g. [level, lang, tweets, phd])
#
#   (*) If the data all have the same label, 
#       create a leaf node which predicts that label, and STOP
#   (*) If the list of attributes is empty (i.e. there are no more possible questions to ask)
#       create a lead node which predicts the most common label, and STOP
#   (*) Otherwise, try partitioning the data by each of the attributes.
#   (*) Choose the partition with the lowest partition entropy.
#   (*) Add a decision node based on the chosen attribute
#   (*) Recur on each partitioned subset using the remaining attributes


# First, we will implement the 'stuff to do' at each node.

# Later, we will set up the recursion logic for the tree-based model


from typing import Dict, TypeVar, Any
from collections import defaultdict

T = TypeVar('T')    # generic type for inputs, e.g. Candidate

# break set into partitions by attribute, represented by a Dict[attribute, partition]
def partitions_by_attribute(inputs: List[T], attribute: str) -> Dict[Any, List[T]]:
    """Partition the inputs into lists based on the specified attribute."""
    partitions: Dict[Any, List[T]] = defaultdict(list)
    for input in inputs:
        key = getattr(input, attribute)   # value of the specified attribute of `input`
        partitions[key].append(input)

    return partitions

# compute partition entropy for a list of partitions
def entropy_by_partition(inputs: List[T], attribute: str, label_attribute: str) -> float:
    """Compute the entropy corresponding to the given artition"""
    # partitions consist of our inputs
    partitions = partitions_by_attribute(inputs, attribute)
    # but partition_entropy needs just the class labels
    labels = [np.array([getattr(input, label_attribute) for input in partition]) for partition in partitions.values()]

    return partition_entropy(labels)


# now, find the minimum entropy partition for the whole dataset:
def min_entropy_partition(inputs: List[T], attributes: List[str], label_attribute: str) -> str:
    entropies = [entropy_by_partition(inputs, attribute, label_attribute)
                for attribute in attributes]
    return attributes[np.argmin(entropies)]

# test
attributes = ['level', 'lang', 'tweets', 'phd']
label_attribute = 'did_well'
for attribute in attributes:
    print(f"{attribute} Entropy: {entropy_by_partition(inputs, attribute, label_attribute)}")
print(f"Min Entropy Partition Attribute: {min_entropy_partition(inputs, attributes, label_attribute)}")


# Now, lets implement the algorithm
# First let's build the tree data type

# A node is either a leaf
class Leaf(NamedTuple):
    value: Any

# or a subtree
class Split(NamedTuple):
    attribute: str
    subtrees: dict
    default_value: Any = None

from typing import Union    # use Union when something could be one of a few types
DecisionTree = Union[Leaf, Split]

def build_tree_id3(inputs: List[Any],
                   split_attributes: List[str],
                   target_attribute: str) -> DecisionTree:
    """Builds subtree at a given node"""
    # count target labels
    label_counts = Counter(getattr(input, target_attribute) for input in inputs)
    most_common_label = label_counts.most_common(1)[0][0] 
    # if there is a unique label, predict it
    if len(label_counts) == 1:
        return Leaf(most_common_label)

    # if no split attributes left, predict most common label
    print(split_attributes)
    if not split_attributes:
        print('bruh')
        return Leaf(most_common_label)
    
    # otherwise, split by attribute with lowest entropy
    # helper function
    def split_entropy(attribute:str) -> float:
        return entropy_by_partition(inputs, attribute, label_attribute)
    
    best_attribute = min(split_attributes, key=split_entropy)

    partitions = partitions_by_attribute(inputs, best_attribute)
    new_attributes = [attribute for attribute in attributes if attribute != best_attribute]

    # recursively build the child trees
    subtrees = {attribute_value : build_tree_id3(subset, new_attributes, label_attribute)
                for attribute_value, subset in partitions.items()}

    return Split(best_attribute, subtrees, default_value=most_common_label)


# Use this to create a model represented by a predict function
def classify(tree: DecisionTree, input: Any) -> Any:
    """classify the input using the given decision tree"""

    # if tree=leaf, return its value
    if isinstance(tree, Leaf):
        return tree.value

    # otherwise, this tree consists of an attribute to split on
    # and a dictionary whose keys are values of that attribute
    # and whose values are subtrees to consider next
    subtree_key = getattr(input, tree.attribute)

    if subtree_key not in tree.subtrees:    # if no subtree for key
        return tree.default_value           # return the default value

    subtree = tree.subtrees[subtree_key]    # Choose the appropriate subtree
    return classify(subtree, input)         # and use it to classify the input


# test
my_tree = build_tree_id3(inputs, attributes, label_attribute)
print('Tree: ', my_tree)
me = Candidate('Junior', 'Python', False, True)
print('Hire?: ', classify(my_tree, me))     # do not hire :(


# NOTE: Big problem
#   With how well the model can fit to the data,
#   it's not a surprise that it is easy to overfit the data
# The Fix:
#   /Random Forests/
#       Build multiple trees (semi-randomly) and combine the results:
#           (*) If output is numerical, we can take an average
#           (*) If output is categorical, we can let them vote
#       How to get randomness?
#           (*) Bootstrap aggregate samples from the inputs
#                   Train each tree on a bootstrap sample
#                   Resulting trees will be pretty different
#           (*) Change the way we choose best_attribute to split on
#                   Rather than looking at all remaining attributes,
#                   Take a random subset and split on the best one of those
#
#   May also consider applying Principal Component Analysis to the set of attributes(?)

#   NOTE: A random forest is an example of a breader technique called /ensemble learning/
#           where we combine several 'weak learners' (high-bias, low-variance)
#           to produce a 'strong learner'

# try to implement later