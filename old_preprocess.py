import spacy
import os
import re
from torchtext.legacy import data

MAX_SAMPLES = 50000
MAX_LENGTH = 40

spacy_en = spacy.load('en_core_web_sm')

tokenize = lambda text : [tok.text for tok in spacy_en.tokenizer(text)]

def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    # adding a start and an end token to the sentence
    return sentence

def load_conversations(path_to_movie_lines, path_to_movie_conversations):
    # dictionary of line id to text
    id2line = {}
    with open(path_to_movie_lines, errors='ignore') as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace('\n', '').split(' +++$+++ ')
        id2line[parts[0]] = parts[4]

    inputs, outputs = [], []
    with open(path_to_movie_conversations, 'r') as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace('\n', '').split(' +++$+++ ')
        # get conversation in a list of line ID
        conversation = [line[1:-1] for line in parts[3][1:-1].split(', ')]
        for i in range(len(conversation) - 1):
            inputs.append(preprocess_sentence(id2line[conversation[i]]))
            outputs.append(preprocess_sentence(id2line[conversation[i + 1]]))
            if len(inputs) >= MAX_SAMPLES:
                return inputs, outputs
    return inputs, outputs


# Tokenize, filter and pad sentences
def tokenize_and_filter(inputs, outputs, start_token, end_token, tokenizer):
    tokenized_inputs, tokenized_outputs = [], []

    for (sentence1, sentence2) in zip(inputs, outputs):
        # tokenize sentence
        print(start_token)
        print(end_token)
        sentence1 = start_token + tokenizer(sentence1) + end_token
        sentence2 = start_token + tokenizer(sentence2) + end_token
        # check tokenized sentence max length
        if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)

    # pad tokenized sentences
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding='post')

    return tokenized_inputs, tokenized_outputs



def get_data():
    path_to_dataset = "cornell movie-dialogs corpus"
    path_to_movie_lines = os.path.join(path_to_dataset, 'movie_lines.txt')
    path_to_movie_conversations = os.path.join(path_to_dataset, 'movie_conversations.txt')

    q, r = load_conversations(path_to_movie_lines, path_to_movie_conversations)

    print(q[26])
    print(r[26])

    # question = data.Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
    # reply = data.LabelField(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
    # fields = { 'question': ('question', question), 'reply': ('reply', reply) }

    words = list(spacy_en.vocab.strings)

    start_token = words[0]
    end_token = words[len(words) - 1]

    questions, replies = tokenize_and_filter(q, r, start_token, end_token, tokenize)

    print('wtf', replies[:, :-1])
    print('wtf1', replies[:, 1:])

    print(fields)

get_data()
