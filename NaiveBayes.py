#
# Naive Bayes Classifiers
#

# A machine learning algorithm suited to boolean features

# Example:
#   We want to determine if a message is spam or not
#   Let S be the event that a message is spam
#
#   Suppose we have n words w_i
#   Let X_i be the event that a message contains the word w_i
#       It assumes values 1, 0 for true, false
#   Let X=(X_1, ..., X_n)
#   
#   We want to calculate P(S | X=x)
#       The probability that a message is spam given that it contains certain words
#
#   Bayes Theorem:
#       P(S | X=x) = ( P(X=x | S) * P(S) ) / ( P(X=x | S) * P(S) + P(X=x | not S) * P(not S) )
#   
#   Assume a message is equally likely to be spam or not spam:
#       P(S | X=x) = P(X=x | S) / ( P(X=x | S) + P(X=x | not S) )
#
#   Naive Bayes Assumption:
#       P(X=x | S) == Product( P(X_i=x_i | S) for all i)
#   
#       This is a very extreme assumption,
#       but the Naive Bayes Classifier often performs well anyway.
#
#   ?????come back???????? Thus it is enough to compute all P(X_i=x_i | S) and P(X_i=x_i | not S)
#       Do this with a 'k-pseudocount':
#           P(X_i=x_i | S) = (k + #(spams containing w_i)) / (2k + #(spams))
#           ????????????? P(X_i=x_i | S) = (k + #(non-spams containing w_i)) / (2k + #(non-spams))
import numpy as np

# First make a function that parses messages into distinct words
from typing import Set
import re

def tokenize(text:str) -> Set[str]:
    text = text.lower()
    all_words = re.findall("[a-z0-9']+", text)  # Find all words consisting of characters a-z, 0-9, '
    return set(all_words)

assert tokenize("Data Science is science") == {'data', 'science', 'is'}

# Define a type for our training data
from typing import NamedTuple

class Message(NamedTuple):
    text: str
    is_spam: bool

# Our classifier needs to keep track of tokens, counts, labels from data
# so we will make it a class
from typing import List, Tuple, Dict, Iterable
import math
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self, k: float=0.5):
        self.k = k  # smoothing/bias factor
        self.tokens: Set[str]=set()
        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        self.token_ham_counts: Dict[str, int] = defaultdict(int)
        self.spam_messages = 0
        self.ham_messages = 0
    
    # Keeps track of numbers used in certain probabilities
    def train(self, messages: Iterable[Message]) -> None:
        for m in messages:
            if m.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1

            for t in tokenize(m.text):
                self.tokens.add(t)
                if m.is_spam:
                    self.token_spam_counts[t] +=1
                else:
                    self.token_ham_counts[t] +=1

    # aux function
    def _probabilities(self, token: str) -> Tuple[float, float]:
        """Returns P(token | spam) and P(token | ham) estimates using pseudocounts"""
        num_spam = self.token_spam_counts[token]
        num_ham = self.token_ham_counts[token]

        p_token_spam = (self.k + num_spam) / (2 * self.k + self.spam_messages)
        p_token_ham = (self.k + num_ham) / (2 * self.k + self.ham_messages)

        return p_token_spam, p_token_ham

    # Predicts if a message is spam using Naive Bayes assumption
    #   Also assuming P(Spam) == P(Ham) == 0.5
    # Use p_1 * ... * p_n = exp( log(p_1) + ... + log(p_n) )
    #   better computational accuracy, prevents 'underflow' (opposite of overflow)
    def predict(self, text: str) -> float:
        text_tokens = tokenize(text)
        log_prob_if_spam = 0.0
        log_prob_if_ham = 0.0

        # iterate thru tokens
        for token in self.tokens:
            prob_if_spam, prob_if_ham = self._probabilities(token)

            # if `token` appears in message, add the log prob of seeing it
            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham += math.log(prob_if_ham)

            # Otherwise add the leg prob of not seeing it
            else:
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_ham += math.log(1.0 - prob_if_ham)
        
        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham = math.exp(log_prob_if_ham)
        return prob_if_spam / (prob_if_spam + prob_if_ham)


# Let's test the model
messages =[Message("spam rules", is_spam=True), Message("ham rules", is_spam=False), Message("hello ham", is_spam=False)]
model = NaiveBayesClassifier()
model.train(messages)

assert model.tokens == {"spam", "ham", "rules", "hello"}
assert model.spam_messages == 1
assert model.ham_messages == 2
assert model.token_spam_counts == {"spam": 1, "rules": 1}
assert model.token_ham_counts == {"ham": 2, "rules": 1, "hello": 1}

text = 'hello spam'

probs_if_spam = [(1 + 0.5) / (1 + 2 * 0.5),      # spam present
                1-(0 + 0.5) / (1 + 2 * 0.5),    # ham not present
                1 - (1 + 0.5) / (1 + 2 * 0.5),  # rules not present
                (0 + 0.5) / (1 + 2 * 0.5)]      # hello present

probs_if_ham = [(0 + 0.5) / (2 + 2 * 0.5),
                1 - (2 + 0.5) / (2 + 2 * 0.5),
                1 - (1 + 0.5) / (2 + 2 * 0.5),
                (1 + 0.5) / (2 + 2 * 0.5)]

p_if_spam = math.exp(sum(math.log(p) for p in probs_if_spam))
p_if_ham = math.exp(sum(math.log(p) for p in probs_if_ham))

assert 0.999 * ( p_if_spam / (p_if_spam + p_if_ham) ) < model.predict(text) < 1.001 * ( p_if_spam / (p_if_spam + p_if_ham) )


# Test on some real data!!!
#   We will use the SpamAssassin public corpus
#       only files prefixed with 20021010

# Download and upack the files
from io import BytesIO  # So we can treat bytes as a file
import requests         # to download the files
import tarfile          # which are in .tar.bz2 format
import tqdm

BASE_URL = "https://spamassassin.apache.org/old/publiccorpus"
FILES = ["20021010_easy_ham.tar.bz2",
         "20021010_hard_ham.tar.bz2",
         "20021010_spam.tar.bz2"]

# This is where the data will end up
OUTPUT_DIR = 'spam_data'

for filename in tqdm.tqdm(FILES):
    # Use requests to get the file contents 
    content = requests.get(f"{BASE_URL}/{filename}").content

    # Wrap the in-memory bytes so we can use them as a "file"
    fin = BytesIO(content)

    # extract all the files to the specified output dir
        #'r:bz2' means read .tar.bz2
    with tarfile.open(fileobj=fin, mode='r:bz2') as tf:
        tf.extractall(OUTPUT_DIR)

# Look through the files
import glob, re, tqdm

# everything 1 folder deep from /spam_data/
#   spam_data/spam/12345.6789
path = 'spam_data/*/*'

data: List[Message] = []

# glob.glob returns every filename: str that matches the wildcarded path
for filename in tqdm.tqdm(glob.glob(path)):
    is_spam = 'ham' not in filename

    # Look through a file in Notepad++ to see 
    #   we will be looking for lines that start with 'Subject:'
    # there are some garbage characters in data
    #   ignore them instead of raising an exception
    with open(filename) as email_file:
        try:
            for line in email_file:
                if line.startswith('Subject:'):
                    subject = line.lstrip('Subject: ').rstrip('\n')
                    data.append(Message(subject, is_spam=is_spam))
                    break
        except (UnicodeDecodeError, PermissionError):
            pass


# Now we split the data into train/test piles
from MachineLearning import split_data, f1_score


train_data, test_data = split_data(data, 0.8)

# And then train a model
model = NaiveBayesClassifier()
model.train(train_data)

from collections import Counter
predictions = [(m, model.predict(m.text)) for m in test_data]

# ~~Assume that spam_probability > 0.5 corresponds to spam prediction~~

# and count the combinations of (actual is_spam, spam_prediction)

from matplotlib import pyplot as plt

xs = [0.01 * n for n in range(101)]
ys = []
for good_enough in tqdm.tqdm(xs):
    confusion_matrix = Counter((m.is_spam, p > good_enough) for (m, p) in predictions)
    try:
        score = f1_score( confusion_matrix[(True, True)],
                confusion_matrix[(False, True)],
                confusion_matrix[(True, False)],
                confusion_matrix[(False, False)] )
    except ZeroDivisionError:
        score = 0
    ys.append(score)

best_cutoff = xs[np.argmax(ys)]
print(f'The Best Cutoff: {100 * best_cutoff}%')

plt.bar(xs, ys, width=0.01)
plt.xlabel('Probability Cutoff')
plt.ylabel('F1-Score of Model')
plt.title('Best Cutoff Probability?')
plt.show()


# See which words are most indicative of spam/ham
def p_spam_given_token(token:str, model: NaiveBayesClassifier) -> float:
    prob_if_spam, prob_if_ham = model._probabilities(token)

    return prob_if_spam / (prob_if_spam + prob_if_ham)

words = sorted(model.tokens, key=lambda t: p_spam_given_token(t, model))

print('Spammiest words:\n', words[-10:])
print("Hammiest words:\n", words[:10])