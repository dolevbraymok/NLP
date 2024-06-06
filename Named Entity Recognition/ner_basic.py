import nltk
from collections import defaultdict, Counter
from nltk.corpus import conll2002

"""# Download necessary NLTK data files
nltk.download('conll2002')
"""
# Load training data (using a different dataset for demonstration)
train_sents = conll2002.iob_sents('esp.train')
test_sents = conll2002.iob_sents('esp.testb')

# Transform the dataset to a word-tag tuple
def transform_to_tuple(sentences):
    return [[(word, tag) for word, pos, tag in sentence] for sentence in sentences]

train_data = transform_to_tuple(train_sents)
test_data = transform_to_tuple(test_sents)

# Compute the most likely tag for each word
word_tag_counts = defaultdict(Counter)

for sentence in train_data:
    for word, tag in sentence:
        word_tag_counts[word][tag] += 1

most_likely_tag = {word: counts.most_common(1)[0][0] for word, counts in word_tag_counts.items()}

# Function to get the most likely tag or NN for unknown words
def get_most_likely_tag(word):
    return most_likely_tag.get(word, 'NN')

# Tag a sentence
def tag_sentence(sentence):
    return [(word, get_most_likely_tag(word)) for word in sentence]

# Evaluate error rate
def evaluate(data):
    total = correct = 0
    for sentence in data:
        words, true_tags = zip(*sentence)
        predicted_tags = [get_most_likely_tag(word) for word in words]
        total += len(true_tags)
        correct += sum(p == t for p, t in zip(predicted_tags, true_tags))
    return 1 - correct / total

# Baseline error rate
baseline_error = evaluate(test_data)
print(f'Baseline error rate: {baseline_error:.4f}')

# Define rules for unknown words
def apply_rules(word):
    if word.istitle():
        return 'NNP'
    if word.isdigit():
        return 'CD'
    if word.endswith('ing'):
        return 'VBG'
    if word.endswith('ed'):
        return 'VBD'
    if word.endswith('s') and len(word) > 1 and not word.endswith('ss'):
        return 'NNS'
    if word.endswith('ly') or word.endswith('ous') or word.endswith('ive') or word.endswith('al'):
        return 'JJ'
    return 'NN'

# Update get_most_likely_tag to use rules for unknown words
def get_most_likely_tag_with_rules(word):
    if word in most_likely_tag:
        return most_likely_tag[word]
    return apply_rules(word)

# Tag a sentence with rules
def tag_sentence_with_rules(sentence):
    return [(word, get_most_likely_tag_with_rules(word)) for word in sentence]

# Evaluate error rate with rules
def evaluate_with_rules(data):
    total = correct = 0
    for sentence in data:
        words, true_tags = zip(*sentence)
        predicted_tags = [get_most_likely_tag_with_rules(word) for word in words]
        total += len(true_tags)
        correct += sum(p == t for p, t in zip(predicted_tags, true_tags))
    return 1 - correct / total

# Error rate with rules
error_with_rules = evaluate_with_rules(test_data)
print(f'Error rate with rules: {error_with_rules:.4f}')


def get_misclassifications(data):
    misclassifications = []
    for sentence in data:
        words, true_tags = zip(*sentence)
        predicted_tags = [get_most_likely_tag_with_rules(word) for word in words]
        for word, true_tag, predicted_tag in zip(words, true_tags, predicted_tags):
            if true_tag != predicted_tag:
                misclassifications.append((word, true_tag, predicted_tag))
    return misclassifications

# Get misclassified words
misclassifications = get_misclassifications(test_data)

# Print some examples of misclassifications
print("Examples of Misclassifications:")
for word, true_tag, predicted_tag in misclassifications[:10]:
    print(f"Word: {word}, True Tag: {true_tag}, Predicted Tag: {predicted_tag}")
