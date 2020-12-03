import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ph_df = pd.read_csv('G:\jw_lab\jwlab_eeg\Scratches\phoneme.csv', header=None)
stimuli = pd.read_csv('G:\jw_lab\jwlab_eeg\Scratches\stimuli.csv', header=None)
ipa_sim_agg_csv = pd.read_csv("G:\jw_lab\jwlab_eeg\\regression\phoneme_data\similarity_aggregated.csv", delimiter='\t')
phonemes = [str.split(s[0]) for s in ph_df.values]
stims = stimuli.values

labels_mapping = {0: 'baby', 1: 'bear', 2: 'bird', 3: 'bunny',
                  4: 'cat', 5: 'dog', 6: 'duck', 7: 'mom',
                  8: 'banana', 9: 'bottle', 10: 'cookie',
                  11: 'cracker', 12: 'cup', 13: 'juice',
                  14: 'milk', 15: 'spoon'}

cmu_phonemes_mapping = {'b': 17, 'k':30, 'd':19, 'm':33, 'j':22, 's':40}  # Corresponding idxs in csv file are two more than here.

first_phonemes_mapping = {'baby':17, 'bear':17, 'bird':17, 'bunny':17,
                                 'cat':30, 'dog':19, 'duck':19, 'mom': 33,
                                 'banana': 17, 'bottle': 17, 'cookie': 30, 'cracker': 30,
                                 'cup': 30, 'juice': 22, 'milk': 33, 'spoon': 40}

stimuli_to_second_ipa_mapping = {'baby':23, 'bear':25, 'bird':24, 'bunny':24,
                                 'cat':16, 'dog':15, 'duck':24, 'mom': 24,
                                 'banana': 24, 'bottle': 15, 'cookie': 37, 'cracker': 39,
                                 'cup': 24, 'juice': 44, 'milk': 25, 'spoon': 38}

# Milk, dog have canadian pronunciation so used that' - mɛlk
# NOTE: 'mom', 'bunny', 'duck' has ʌ as a ipa character and is not present in the .csv file.
# 'cookie' has ʊ character which is not present. Replaced with ɔ.
# Note: 'bird', 'mom' not present -> chosen closest sounding alternative. Corresponding idxs in csv file are two more than here.


phonemes_list = [item for sublist in phonemes for item in sublist]
phoneme_set = list(set(phonemes_list))


# First define the phonemes as per IPA format.
# NOTE: The labels_mapping dictionary contains the stimuli.

def from_sim_agg_first_phoneme():
    # This function creates an .npz of the phoneme vectors corresponding to the similarity_aggregated.csv file.
    phoneme_sim_agg_first_embeds = {}
    for i, stim in enumerate(stims):
        ph = stim[0]
        ph_mapping_idx = first_phonemes_mapping[ph]
        if ph_mapping_idx is not None:
            phoneme_sim_agg_first_embeds[i] = ipa_sim_agg_csv.iloc[ph_mapping_idx, 1:].values

    dict_array = []
    dict_array.append(phoneme_sim_agg_first_embeds)
    return dict_array


def from_sig_agg_second_phoneme():
    # This function creates an .npz of the second phoneme vectors corresponding to the similarity_aggregated.csv file.
    phoneme_sim_agg_second_embeds = {}
    for i, stim in enumerate(stims):
        ph = stim[0]
        ph_mapping_idx = stimuli_to_second_ipa_mapping[ph]
        if ph_mapping_idx is not None:
            phoneme_sim_agg_second_embeds[i] = ipa_sim_agg_csv.iloc[ph_mapping_idx, 1:].values

    dict_array = []
    dict_array.append(phoneme_sim_agg_second_embeds)
    return dict_array

def create_ph_embeds():
    ph_embeds = np.zeros((len(stims), len(phoneme_set)))
    # First iterate over the phonemes.
    for i, ph_word in enumerate(phonemes):
        # for ph in ph_word:
        ph = ph_word[0]
        ind = phoneme_set.index(ph)
        ph_embeds[i, ind] = 1

    return ph_embeds


def create_ph_one_hots_first():
    first_ph_set = []
    for i, ph_word in enumerate(phonemes):
        ph = ph_word[1]
        if ph not in first_ph_set:
            first_ph_set.append(ph)
    ph_embeds = np.zeros((len(stims), len(first_ph_set)))
    # First iterate over the phonemes.
    for i, ph_word in enumerate(phonemes):
        # for ph in ph_word:
        ph = ph_word[1]  # The second phoneme.
        ind = first_ph_set.index(ph)
        ph_embeds[i, ind] = 1

    return ph_embeds


def create_ph_classes():
    first_ph_set = []
    for i, ph_word in enumerate(phonemes):
        ph = ph_word[1]
        if ph not in first_ph_set:
            first_ph_set.append(ph)
    first_ph_list = list(first_ph_set)
    classes = []
    for i, ph_word in enumerate(phonemes):
        ph = ph_word[1]
        ind = first_ph_list.index(ph)
        classes.append(ind)

    return classes



sim_agg_embeddings = from_sim_agg_first_phoneme()
np.savez_compressed("G:\\jw_lab\\jwlab_eeg\\regression\\phoneme_embeddings\\first_sim_agg_embeddings.npz", sim_agg_embeddings)


# second_sim_agg_embeddings = from_sig_agg_second_phoneme()
# np.savez_compressed("G:\\jw_lab\\jwlab_eeg\\regression\\phoneme_embeddings\\second_sim_agg_embeddings.npz", second_sim_agg_embeddings)

# classes = create_ph_classes()
# np.savez_compressed("G:\\jw_lab\\jwlab_eeg\\regression\\phoneme_embeddings\\second_phoneme_classes.npz", classes)

# ph_embeds = create_ph_embeds()


# np.savez_compressed("G:\\jw_lab\\jwlab_eeg\\regression\\phoneme_embeddings\\phoneme_embeds.npz", ph_embeds)

# first_one_hots = create_ph_one_hots_first()
# np.savez_compressed("G:\\jw_lab\\jwlab_eeg\\regression\\phoneme_embeddings\\second_one_hots.npz", first_one_hots)