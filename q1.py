import numpy
from nltk.translate import AlignedSent, Alignment
import json
import all_alignment
import copy
import timeit
import nltk
#can alternatively use itertools instead of align_funct to gen all possible permutaions
import itertools
#stores all possible alignments for a translation pair
permutations_temp = []

#recursive function for generation all possible permuations based on back-tracking
def permutations_helper(array, data, last, index):
    length = len(array)
    for i in range(length):
        data[index] = array[i]
        if index == last:
            permutations_temp.append(copy.copy(data))
        else:
            permutations_helper(array, data, last, index + 1)

def permutations(array, length):
    data = [0] * (length)
    permutations_helper(array, data, length - 1, 0)

#returns all possible alignments
def align_funct(f_size, english_size):
    global permutations_temp
    permutations_temp = []
    english_index = []
    f_index = []
    for i in range(english_size):
        english_index.append(i + 1)
    for i in range(f_size):
        f_index.append(i + 1)
    permutations(english_index, f_size)
    all_alignm = []
    for perm in permutations_temp:
        temp_perm = []
        for i, j in zip(f_index, perm):
            temp_perm.append((i, j))
        all_alignm.append(temp_perm)
    return(all_alignm)

def helper_dups(list):
    final_list = []
    for word in list:
        if word not in final_list:
            final_list.append(word)
    return final_list

def vocabulary_indexing(e_sentences, f_sentences):
    eng_dict = []
    foreign_dict = []
    for e_sentence, f_sentence in zip(e_sentences, f_sentences):
        for e_word in e_sentence.split():
            eng_dict.append(e_word)
        for f_word in f_sentence.split():
            foreign_dict.append(f_word)
    return helper_dups(eng_dict), helper_dups(foreign_dict)

#calculates translation probabilities for each word pair
def translation_model(eng_dict, foreign_dict, e_sentences, f_sentences, iter):
#initially all pairs have equal chances of being translated
    t = numpy.full((len(eng_dict), len(foreign_dict)), 1.0 / (len(eng_dict)))
#with each iteration as the alignemnet probs change, word translation probs also change
    for iter in range(iter):
        c = numpy.full((len(eng_dict), len(foreign_dict)), 0.0)
        total = numpy.full((len(foreign_dict),), 0.0)
        s_t = numpy.full((len(eng_dict),), 0.00001)
        for e_sentence, f_sentence in zip(e_sentences, f_sentences):
            for e_word in e_sentence.split():
                for f_word in f_sentence.split():
                    s_t[eng_dict.index(e_word)] = s_t[eng_dict.index(e_word)] + t[eng_dict.index(e_word)][foreign_dict.index(f_word)]
            for e_word in e_sentence.split():
                for f_word in f_sentence.split():
                    c[eng_dict.index(e_word)][foreign_dict.index(f_word)] = c[eng_dict.index(e_word)][foreign_dict.index(f_word)] + t[eng_dict.index(e_word)][foreign_dict.index(f_word)] / s_t[eng_dict.index(e_word)]
                    total[foreign_dict.index(f_word)] = total[foreign_dict.index(f_word)] + t[eng_dict.index(e_word)][foreign_dict.index(f_word)] / s_t[eng_dict.index(e_word)]
        for e_word in eng_dict:
            for f_word in foreign_dict:
                t[eng_dict.index(e_word)][foreign_dict.index(f_word)] = c[eng_dict.index(
                    e_word)][foreign_dict.index(f_word)] / total[foreign_dict.index(f_word)]
    return t

#calc prob for an alignment
def alignments_model(e_sentence, f_sentence, eng_dict, foreign_dict, alignment, t):
    prob = 1.0
    for e_word in e_sentence:
        sum = 0.0
        for f_word in f_sentence:
            sum += t[eng_dict.index(e_word)][foreign_dict.index(f_word)]
        if alignment[e_sentence.index(e_word)][1] != 0:
            prob = prob * (t[eng_dict.index(e_word)][foreign_dict.index(f_sentence[alignment[e_sentence.index(e_word)][1] - 1])] / sum)
    return prob

#returns final ans for each sent pair in corpus
def get_best_alignment(e_sentence, f_sentence, eng_dict, foreign_dict, t):
    e_sentence = e_sentence.split()
    f_sentence = f_sentence.split()
    max_prob = 0.0
    best_align = []
    alignments = align_funct(len(e_sentence), len(f_sentence))
    for alignment in alignments:
        prob = alignments_model(e_sentence, f_sentence, eng_dict, foreign_dict, alignment, t)
        if prob > max_prob:
            max_prob = prob
            best_align = alignment
    return best_align

#with open('data1.json') as f:
with open('own.json') as f:
	data = json.load(f)
f_sentences = []
e_sentences = []
for pair in data:
	#f_sentences.append(pair['fr'].lower())
	f_sentences.append(pair['gn'].lower())
	e_sentences.append(pair['en'].lower())
#start_time = timeit.default_timer()
#build separate vocabulary for each language
eng_dict, foreign_dict = vocabulary_indexing(e_sentences, f_sentences)
t = translation_model(eng_dict, foreign_dict, e_sentences, f_sentences, 50)
final_ans = []
for e_sentence, f_sentence in zip(e_sentences, f_sentences):	
	best_align = get_best_alignment(e_sentence, f_sentence, eng_dict, foreign_dict, t)
	final_ans.append((e_sentence, f_sentence, best_align))
#print(timeit.default_timer() - start_time)
for ans in final_ans:
	print(ans)
