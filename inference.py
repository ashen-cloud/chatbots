#!/opt/homebrew/bin/python3

import torch
from torchtext.data.utils import get_tokenizer
import tensorflow as tf # todo: get rid of this
from preprocess import text_transform, MAX_LENGTH, device

model = torch.load('trained_model', map_location=torch.device(device))

voc = torch.load('vocab')

print('accepting input')
while True:
    query = input()

    q_tokenized = text_transform(query, voc, get_tokenizer('basic_english'))

    q_padded = tf.keras.preprocessing.sequence.pad_sequences([q_tokenized], maxlen=MAX_LENGTH, padding='post', value=1.0)

    q_tensor = torch.Tensor(q_padded).int().to(device)

    infer = model(q_tensor)[0]

    int_infer = infer.int()

    tokens = [t.item() for t in int_infer[0]]
    words = voc.get_itos()
    result = ' '.join(list(filter(lambda w: '<' not in w and '>' not in w, [words[t] for t in tokens])))
    print(result)



