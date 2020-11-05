"""
Functions in the same manner as align.py using nltk implementation of IBM models.
"""
from nltk.translate import AlignedSent, Alignment
from nltk import IBMModel1
from nltk import IBMModel2
import json
import timeit


def init_sent():
    all_sent = []

    with open('data2.json') as f:
        data = json.load(f)

    f_sentences = []
    e_sentences = []

    for pair in data:
        f_sentences.append(pair['fr'].lower())
        e_sentences.append(pair['en'].lower())

    for e_sentence, f_sentence in zip(e_sentences, f_sentences):
        f_sentence = f_sentence.split()
        e_sentence = e_sentence.split()
        all_sent.append(AlignedSent(f_sentence, e_sentence))

    return all_sent


def print_alignment(all_sent):

    for sent in all_sent:
        print(sent.words)
        print(sent._mots)
        print(sent.alignment)


if __name__ == "__main__":
    
    start_time = timeit.default_timer()

    all_sent = init_sent()

    IBMModel1(all_sent, 50)
    print("By IMB model1 ,this is the output\n")
    print(all_sent)
    print("\n")
    #print_alignment(all_sent)
    print(timeit.default_timer() - start_time)
    start_time = timeit.default_timer()
    print("\n")

    all_sent = init_sent()
    IBMModel2(all_sent, 50)
    print("By IBM model2 ,this is the output\n")
    print(all_sent)

    #print_alignment(all_sent)
    print(timeit.default_timer() - start_time)
    print("\n")
