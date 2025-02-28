import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import matplotlib.pyplot as plt
import string
import numpy as np
labels_mapping = {0: 'baby', 1: 'bear', 2: 'bird', 3: 'bunny',
                  4: 'cat', 5: 'dog', 6: 'duck', 7: 'mum',
                  8: 'banana', 9: 'bottle', 10: 'cookie',
                  11: 'cracker', 12: 'cup', 13: 'juice',
                  14: 'milk', 15: 'spoon'}



# Load the data.
data = np.load('/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/cbt_childes_data.npz', allow_pickle=True)
data = data['arr_0']
data = data.tolist()


# Model fine-tuning.
model_2 = Word2Vec(size=300, min_count=1, sg=1, workers=16)
model_2.build_vocab(data)
total_examples = model_2.corpus_count


model_2.intersect_word2vec_format("/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/w2v_models/GoogleNews-vectors-negative300.bin", binary=True, lockf=1.0)

# Initiate the fine tuning process.
print('checking.')
iters = 1000
model_2.train(data, total_examples=total_examples, epochs=iters)#, start_alpha=0.1, end_alpha=0.025)

# Save the model.
model_2.wv.save_word2vec_format("/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/w2v_models/tuned_vectors_pre_w2v_cbt_childes_1.bin", binary=True)

# # model = KeyedVectors.load_word2vec_format("G:\\jw_lab\\jwlab_eeg\\regression\\GoogleNews-vectors-negative300.bin.gz", binary=True)

# # children_stories read-in
# children_stories_train = pd.read_fwf('G:\jw_lab\jwlab_eeg\Scratches\data\cbt_train.txt', header=None)
# children_stories_test = pd.read_fwf('G:\jw_lab\jwlab_eeg\Scratches\data\cbt_test.txt', header=None)
# children_stories_valid = pd.read_fwf('G:\jw_lab\jwlab_eeg\Scratches\data\cbt_valid.txt', header=None)

# children_stories_train = children_stories_train.iloc[:, 0]
# children_stories_test = children_stories_test.iloc[:, 0]
# children_stories_valid = children_stories_valid.iloc[:, 0]


# def get_childes_treebank():
#     hslld_er = pd.read_csv('G:\jw_lab\jwlab_eeg\Scratches\data\hslld-hv1-er.csv')
#     hslld_mt = pd.read_csv('G:\jw_lab\jwlab_eeg\Scratches\data\hslld-hv1-mt.csv')

#     childes_treebank_df = pd.concat([hslld_er, hslld_mt], axis=0).iloc[:,1]
#     childes_treebank_df = childes_treebank_df.to_list()
#     return childes_treebank_df




# def get_childes_freq():
#     stim_freq_dict = {}
#     for label in labels_mapping.values():
#         stim_freq_dict[label] = 0
#     stim_freq_dict['mummy'] = 0

#     with open('G:\jw_lab\jwlab_eeg\Scratches\data\hslld-hv1-er.txt') as f:
#         freq_dict = Counter(f.read().lower().translate(str.maketrans('', '', string.punctuation)).split())
#         for stim in labels_mapping.values():
#             all_stim_variants = get_all_word_variants(stim)  # Returns a dictionary.
#             variants = all_stim_variants[stim]  # Extracting the list of variants.
#             for variant in variants:
#                 stim_freq_dict[stim] += freq_dict[variant]
#         stim_freq_dict['mummy'] = freq_dict['mummy']

#     with open('G:\jw_lab\jwlab_eeg\Scratches\data\hslld-hv1-mt.txt') as f:
#         freq_dict = Counter(f.read().lower().translate(str.maketrans('', '', string.punctuation)).split())
#         for stim in labels_mapping.values():
#             all_stim_variants = get_all_word_variants(stim)  # Returns a dictionary.
#             variants = all_stim_variants[stim]  # Extracting the list of variants.
#             for variant in variants:
#                 stim_freq_dict[stim] += freq_dict[variant]
#         stim_freq_dict['mummy'] = freq_dict['mummy']
#     return stim_freq_dict


# def get_all_word_variants(stim):
#     # Get all forms of a word. Eg.- capitalized, all caps, plurals, and lowercase.
#     variant_list = {}
#     plural_dict = {'baby':'babies', 'bear':'bears', 'bird': 'birds', 'bunny':' bunnies', 'cat': 'cats',
#                    'dog':'dogs', 'duck':'ducks', 'banana':'bananas', 'bottle': 'bottles', 'cookie': ' cookies',
#                    'cracker': 'crackers', 'cup': 'cups', 'juice': 'juices', 'spoon':'spoons'}
#                 # NOTE: 'mum' and 'milk' are not present here.
#     if stim == "mum":
#         variant_list[stim] = []
#         # Uppercase
#         variant_list[stim].append(stim.upper())
#         #lowercase
#         variant_list[stim].append(stim)  # stim is already in lowercase
#         # First letter capitalized.
#         variant_list[stim].append(stim.title())
#     else:
#         variant_list[stim] = []
#         variant_list[stim].append(stim.upper())
#         # lowercase
#         variant_list[stim].append(stim)  # stim is already in lowercase
#         # First letter capitalized.
#         variant_list[stim].append(stim.title())
#         # Plural
#         if stim != 'milk':
#             variant_list[stim].append(plural_dict[stim])

#     return variant_list




# def get_freq_words():
#     stim_freq_dict = {}
#     for label in labels_mapping.values():
#         stim_freq_dict[label] = 0

#     with open('G:\jw_lab\jwlab_eeg\Scratches\data\cbt_train.txt') as f:
#         freq_dict = Counter(f.read().lower().split())
#         for stim in labels_mapping.values():
#             all_stim_variants = get_all_word_variants(stim)  # Returns a dictionary.
#             variants = all_stim_variants[stim]  # Extracting the list of variants.
#             for variant in variants:
#                 stim_freq_dict[stim] += freq_dict[variant]
#                 # if stim != 'mom':
#                 #     stim_freq_dict[stim] += freq_dict[variant]
#                 # else:
#                 #     stim_freq_dict['mum'] += freq_dict[variant]

#     with open('G:\jw_lab\jwlab_eeg\Scratches\data\cbt_test.txt') as f:
#         freq_dict = Counter(f.read().lower().split())
#         for stim in labels_mapping.values():
#             all_stim_variants = get_all_word_variants(stim)  # Returns a dictionary.
#             variants = all_stim_variants[stim]  # Extracting the list of variants.
#             for variant in variants:
#                 stim_freq_dict[stim] += freq_dict[variant]
#                 # if stim != 'mom':
#                 #     stim_freq_dict[stim] += freq_dict[variant]
#                 # else:
#                 #     stim_freq_dict['mum'] += freq_dict[variant]

#     with open('G:\jw_lab\jwlab_eeg\Scratches\data\cbt_valid.txt') as f:
#         freq_dict = Counter(f.read().lower().split())
#         for stim in labels_mapping.values():
#             all_stim_variants = get_all_word_variants(stim)  # Returns a dictionary.
#             variants = all_stim_variants[stim]  # Extracting the list of variants.
#             for variant in variants:
#                 stim_freq_dict[stim] += freq_dict[variant]
#                 # if stim != 'mom':
#                 #     stim_freq_dict[stim] += freq_dict[variant]
#                 # else:
#                 #     stim_freq_dict['mum'] += freq_dict[variant]
#     return stim_freq_dict


# def get_w2v_word_freq():
#     stim_freq_dict = {}
#     # for label in labels_mapping.values():
#     #     stim_freq_dict[label] = 0
#     #
#     for label in labels_mapping.values():
#         word_count = model.vocab[label].count
#         stim_freq_dict[label] = word_count
#     stim_freq_dict['mom'] = model.vocab['mom'].count

#     return stim_freq_dict

# stim_freqs = get_w2v_word_freq()

# # Plot histogram of words.
# plt.clf()
# plt.bar(stim_freqs.keys(), stim_freqs.values(), color='c')
# for i, v in enumerate(list(stim_freqs.values())):
#     plt.text(i, v, str(v), ha='center', color='r')
# plt.xticks(rotation=270)
# plt.title("Distribution of stimuli words in W2V")
# plt.xlabel("Word")
# plt.ylabel("Frequency")
# plt.show()



# # Remove book titles and preprocess data.
# sentences_only_all_sets = []
# for row in children_stories_train:
#     if ("_BOOK_" not in row) and ("CHAPTER" not in row):
#         sentences_only_all_sets.append(row)
# for row in children_stories_test:
#     if ("_BOOK_" not in row) and ("CHAPTER" not in row):
#         sentences_only_all_sets.append(row)
# for row in children_stories_valid:
#     if ("_BOOK_" not in row) and ("CHAPTER" not in row):
#         sentences_only_all_sets.append(row)



# tokenizer = RegexpTokenizer(r'\w+')

# # For the CB_test data
# sentences_tokenized = [w.lower() for w in sentences_only_all_sets]
# sentences_tokenized = [tokenizer.tokenize(i) for i in sentences_tokenized]


# # Process childes treebank data.
# childes_sentences = get_childes_treebank()
# sentences_tokenized_childes = [w.lower() for w in childes_sentences]
# sentences_tokenized_childes = [tokenizer.tokenize(i) for i in sentences_tokenized_childes]


# # Save the concatenated list to avoid preprocessing again.

# cbt_childes_data = []
# cbt_childes_data.extend(sentences_tokenized)
# cbt_childes_data.extend(sentences_tokenized_childes)
# cbt_childes_data = np.array(cbt_childes_data)
# np.savez_compressed('/regression/cbt_childes_data.npz', cbt_childes_data)

# model_2 = Word2Vec(size=300, min_count=1, sg=1)
# model_2.build_vocab(sentences_tokenized)
# total_examples = model_2.corpus_count

# # Training only w2v_cbt_full_skipgram.
# # iters = 50
# # model_2.train(sentences_tokenized, total_examples=total_examples, epochs=iters, compute_loss=True)
# # model_2.wv.save_word2vec_format("w2v_cbtest_full_only_skipgram.bin", binary=True)
# # The following two lines are for fine tuning (intersecting the existing model with the new one).

# model = KeyedVectors.load_word2vec_format("G:\jw_lab\jwlab_eeg\\tuned_w2v_pre_childes_treebank.bin", binary=True)

# model_2.build_vocab([list(model.vocab.keys())], update=True)

# model_2.intersect_word2vec_format("G:\jw_lab\jwlab_eeg\\tuned_w2v_pre_childes_treebank.bin", binary=True)

# # Initiate the fine tuning process.
# iters = 50
# training_return = model_2.train(sentences_tokenized, total_examples=total_examples, epochs=iters, compute_loss=True)


# # model_2.wv.save_word2vec_format("tuned_w2v_pre_cbt.bin", binary=True)
# model_2.wv.save_word2vec_format("w2v_pre_childes_cbt_tuned.bin", binary=True)
# # tuned_w2v = KeyedVectors.load_word2vec_format("tuned_w2v_pre_cbt.bin", binary=True)


# print(tuned_w2v.similarity("hello", "bro"))

# # I have two modified versions of Word2Vec.
# # 'tuned_w2v' is the fine-tuned on the CBTest dataset.
# # 'w2v_cbtest_only.bin' is the Word2Vec model trained only on the CBTest dataset.