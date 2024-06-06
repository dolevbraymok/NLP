import nltk
import random
from collections import defaultdict, Counter
from nltk.corpus import conll2002


#Download if not installed:
"""nltk.download('conll2002')
"""


def transform_to_tuple(sentences):
    """
    Function change the sentences into an array of tuples: (Word,tag) (tag of Penn Treebank tagset)
    :param sentences:
    :return: array of words and tag of each sentence
    Example: return[i] is sentence i and return[i][j] is (word, tag) of word j in it
    """
    return [[(word, tag) for word, pos, tag in sentence] for sentence in sentences]


# Train the HMM
def train_hmm(train_set):
    """
    Function that creates a Hidden Markov Model (HMM) based on the input training set.

    This function calculates the transition probabilities, emission probabilities, and tag counts
    from the training data. These probabilities form the basis of the HMM, which can be used
    for tasks such as part-of-speech tagging or named entity recognition.

    :param train_set: List of sentences, where each sentence is a list of (word, tag) tuples.
                      Example: [[('The', 'DET'), ('dog', 'NOUN')], [('barks', 'VERB')]]
    :return: Three dictionaries:
             - transition_probs: Dictionary of transition probabilities between tags.
               transition_probs[x][y] = P(tag_i == y | tag_i-1 == x)
             - emission_probs: Dictionary of emission probabilities of words given tags.
               emission_probs[x][y] = P(word_i == y | tag_i == x)
             - tag_counts: Dictionary of normalized counts of tags.
               tag_counts[x] = occurrences of tag x / total occurrences of all tags
    """
    transition_probs = {}  # transition probabilities: P(tag_i | tag_i-1)
    emission_probs = {}  # emission probabilities: P(word | tag)
    tag_counts = {}  # counts of each tag

    for sentence in train_set:
        prev_tag = None
        for word, tag in sentence:
            # Update tag counts
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

            # Update transition probabilities
            if prev_tag is not None:
                if prev_tag not in transition_probs:
                    transition_probs[prev_tag] = {}
                transition_probs[prev_tag][tag] = transition_probs[prev_tag].get(tag, 0) + 1
            prev_tag = tag

            # Update emission probabilities
            if tag not in emission_probs:
                emission_probs[tag] = {}
            emission_probs[tag][word] = emission_probs[tag].get(word, 0) + 1

    # Normalize probabilities
    for prev_tag in transition_probs:
        total_transitions = sum(transition_probs[prev_tag].values())
        for tag in transition_probs[prev_tag]:
            transition_probs[prev_tag][tag] /= total_transitions

    for tag in emission_probs:
        total_emissions = sum(emission_probs[tag].values())
        for word in emission_probs[tag]:
            emission_probs[tag][word] /= total_emissions

    # Normalize tag counts
    total_tags = sum(tag_counts.values())
    for tag in tag_counts:
        tag_counts[tag] /= total_tags

    return transition_probs, emission_probs, tag_counts


# Implement the Viterbi algorithm
def viterbi(obs, states, transition_probs, emission_probs, tag_counts):
    """
    Implementation of the Viterbi algorithm for Named Entity Recognition (NER) problem.

    The Viterbi algorithm finds the most probable sequence of hidden states (tags) given
    a sequence of observed events (words) and a model of the transition probabilities
    between states and emission probabilities of observations given states.

    :param obs: List of observations (e.g., words in a sentence).
    :param states: List of possible states (e.g., NER tags).
    :param transition_probs: Dictionary of transition probabilities between states.
                             transition_probs[from_state][to_state] = P(to_state | from_state)
    :param emission_probs: Dictionary of emission probabilities of observations given states.
                           emission_probs[state][observation] = P(observation | state)
    :param tag_counts: Dictionary of initial probabilities of states.
                       tag_counts[state] = P(state)

    :return: A tuple containing the probability of the most probable state sequence and
             the most probable state sequence itself as a list.
             (prob, path[state])
             prob: Probability of the most probable state sequence.
             path[state]: List of states representing the most probable state sequence.
    """
    V = [{}]
    path = {}

    # Initialize base case (t == 0)
    for state in states:
        V[0][state] = tag_counts[state] * emission_probs[state].get(obs[0], 0)
        path[state] = [state]

    # Recurrence relation
    for t in range(1, len(obs)):
        V.append({})
        new_path = {}

        for state in states:
            (prob, prev_state) = max((V[t - 1][prev_state] * transition_probs[prev_state].get(state, 0) *
                                      emission_probs[state].get(obs[t], 0), prev_state)
                                      for prev_state in states)
            V[t][state] = prob
            new_path[state] = path[prev_state] + [state]

        # Update path
        path = new_path

    # Termination
    (prob, state) = max((V[len(obs) - 1][state], state) for state in states)

    return prob, path[state]

# Decode test sentences
def decode_test_sentences(test_set, transition_probs, emission_probs, tag_counts):
    """
    Decodes a set of test sentences using the Viterbi algorithm and calculates the accuracy of the predicted tags.

    This function iterates over each sentence in the test set, uses the Viterbi algorithm to predict the sequence of
    tags for the observed words, and then compares the predicted tags to the true tags to calculate the overall accuracy.

    :param test_set: List of sentences, where each sentence is a list of (word, true_tag) tuples.
    :param transition_probs: Dictionary of transition probabilities between states (tags).
                             transition_probs[from_state][to_state] = P(to_state | from_state)
    :param emission_probs: Dictionary of emission probabilities of observations (words) given states (tags).
                           emission_probs[state][observation] = P(observation | state)
    :param tag_counts: Dictionary of initial probabilities of states (tags).
                       tag_counts[state] = P(state)

    :return: The accuracy of the predicted tags as a float value, which is the ratio of correctly predicted tags
             to the total number of words in the test set.
    """
    total_words = 0
    correct_tags = 0

    for sentence in test_set:
        obs = [word for word, _ in sentence]
        true_tags = [tag for _, tag in sentence]
        states = emission_probs.keys()
        _, predicted_tags = viterbi(obs, states, transition_probs, emission_probs, tag_counts)

        for true_tag, predicted_tag in zip(true_tags, predicted_tags):
            total_words += 1
            if true_tag == predicted_tag:
                correct_tags += 1

    return correct_tags / total_words



train_sents = conll2002.iob_sents('esp.train')
test_sents = conll2002.iob_sents('esp.testb')


train_set = transform_to_tuple(train_sents)
test_set = transform_to_tuple(test_sents)
transition_probs, emission_probs, tag_counts = train_hmm(train_set)
# Evaluate performance
error_rate = 1 - decode_test_sentences(test_set, transition_probs, emission_probs, tag_counts)
print(f'HMM Tagger Error Rate: {error_rate:.4f}')
