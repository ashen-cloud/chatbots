import os
import re
import tensorflow as tf
import tensorflow_datasets as tfds
from collections import Counter
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch

MAX_SAMPLES = 50000
MAX_LENGTH = 40

unk_token = '<unk>'
pad_token = '<PAD>'
bos_token = '<BOS>'
eos_token = '<EOS>'

unk_token_ind = 0
pad_token_ind = 1
bos_token_ind = 2
eos_token_ind = 3

def create_dataloader():
    # TODO: remove tensorflow
    path_to_zip = tf.keras.utils.get_file('cornell_movie_dialogs.zip',
                                          origin='http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip',
                                          extract=True)
    path_to_dataset = os.path.join(os.path.dirname(path_to_zip), "cornell movie-dialogs corpus")
    path_to_movie_lines = os.path.join(path_to_dataset, 'movie_lines.txt')
    path_to_movie_conversations = os.path.join(path_to_dataset, 'movie_conversations.txt')

    questions, answers = load_conversations(path_to_movie_lines, path_to_movie_conversations)

    tokenizer = get_tokenizer('basic_english')

    counter = Counter()
    for sent in questions + answers:
        counter.update(tokenizer(sent))

    voc = vocab(counter)
    voc.insert_token(token=unk_token, index=unk_token_ind)
    voc.set_default_index(index=unk_token_ind)
    voc.insert_token(token=pad_token, index=pad_token_ind)
    voc.insert_token(token=bos_token, index=bos_token_ind)
    voc.insert_token(token=eos_token, index=eos_token_ind)

    text_transform = lambda x: [voc['<BOS>']] + [voc[token] for token in tokenizer(x)] + [voc['<EOS>']]

    print(text_transform(questions[0]))

    q_tokenized = [text_transform(t) for t in questions]
    a_tokenized = [text_transform(t) for t in answers]

    q_padded = tf.keras.preprocessing.sequence.pad_sequences(
        q_tokenized, maxlen=MAX_LENGTH, padding='post', value=1.0)

    a_padded = tf.keras.preprocessing.sequence.pad_sequences(
        a_tokenized, maxlen=MAX_LENGTH, padding='post', value=1.0)

    print("Vocab len", len(voc))

    dataloader = DataLoader(list(zip(q_padded, a_padded)), batch_size=8, shuffle=False)
    # dataloader = list(iter(dl_inst))

    return dataloader, text_transform, voc


def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    return sentence


def load_conversations(path_to_movie_lines, path_to_movie_conversations):
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
        conversation = [line[1:-1] for line in parts[3][1:-1].split(', ')]
        for i in range(len(conversation) - 1):
            inputs.append(preprocess_sentence(id2line[conversation[i]]))
            outputs.append(preprocess_sentence(id2line[conversation[i + 1]]))
            if len(inputs) >= MAX_SAMPLES:
                return inputs, outputs
    return inputs, outputs
