from sklearn.decomposition import PCA, TruncatedSVD
import numpy as np
import pandas as pd
import os
import pickle
from functions import load_llm_embeds, get_transformer_embeddings_from_dict
import matplotlib.pyplot as plt

import matplotlib.cm as cm

labels_mapping = {0: 'baby', 1: 'bear', 2: 'bird', 3: 'bunny',
                  4: 'cat', 5: 'dog', 6: 'duck', 7: 'mom',
                  8: 'banana', 9: 'bottle', 10: 'cookie',
                  11: 'cracker', 12: 'cup', 13: 'juice',
                  14: 'milk', 15: 'spoon'}

PATH_MAPPING = {
    'gpt2-large': '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/llm_embeds/gpt2-large_all_words_embeddings_layer_wise.pkl',
    'gpt2-xl': '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/llm_embeds/gpt2-xl_all_words_embeddings_layer_wise.pkl',
    'w2v': '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/w2v_embeds/embeds_with_label_dict.npz'
}


NUM_LAYER_MAPPING = {
    'gpt2-large': 37,
    'gpt2-xl': 48,
}

STORE_PATH_MAPPING = {
    'gpt2-large': '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/llm_embeds/',
    'gpt2-xl': '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/llm_embeds/',
    'w2v': '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/jwlab_eeg/regression/w2v_embeds/',
}

model_type = 'gpt2-large'
n_components = 300

# First ensure that the vectors is of shape (n_samples, n_features).

labels = list(labels_mapping.keys())

if model_type == 'w2v':
    embeds_with_labels_dict_loaded = np.load(PATH_MAPPING[model_type], allow_pickle=True)
    embeds = embeds_with_labels_dict_loaded['arr_0']
    import pdb; pdb.set_trace()
    embeds = embeds[0]
    for label in labels:
        embeds = embeds[label]
    embeds = np.array(embeds)

    # Apply the SVD and then create a dictionary mapping the labels to the reduced embeddings.
    svd = TruncatedSVD(n_components=n_components)
    embeds = svd.fit_transform(embeds)
    
    # Print the number explained variance.
    print('Explained variance: ')
    print(svd.explained_variance_ratio_)

    # Create a dictionary mapping the labels to the reduced embeddings.
    embeds_dict = {label: embeds[i] for i, label in enumerate(labels)}

    # Save the dictionary.
    with open(os.path.join(STORE_PATH_MAPPING[model_type], 'embeds_with_label_dict_reduced_svd.pkl'), 'wb') as f:
        pickle.dump(embeds_dict, f)

else:
    embeds = load_llm_embeds(model_type)  # This is a dictionary.
    
    # Create a larger figure for better visualization
    plt.figure(figsize=(15, 10))
    
    # Store explained variance ratios for each layer
    layer_variances = []
    
    # Create a colormap for the gradient effect
    colormap = cm.viridis  # You can try other colormaps: 'plasma', 'Blues', 'YlOrRd', etc.
    num_layers = NUM_LAYER_MAPPING[model_type]
    
    for layer in range(num_layers):
        layer_labels = get_transformer_embeddings_from_dict(labels, embeds_dict=embeds, layer=layer)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        layer_labels_svd = svd.fit_transform(layer_labels)
        
        # Store the explained variance ratio for this layer
        layer_variances.append(svd.explained_variance_ratio_)
        print(f'Layer {layer}: {svd.explained_variance_ratio_}')
        
        # Calculate color based on layer number (normalize to 0-1 range)
        color = colormap(layer / (num_layers - 1))
        
        # Plot the cumulative explained variance for this layer
        x_values = range(1, len(svd.explained_variance_ratio_) + 1)
        y_values = np.cumsum(svd.explained_variance_ratio_)
        line, = plt.plot(x_values, y_values, 
                 color=color,
                 alpha=0.7, 
                 label=f'Layer {layer}')
        
        # Add text label at the beginning of the line
        plt.annotate(f'{layer}', 
                    xy=(x_values[0], y_values[0]),
                    xytext=(-10, 0),  # Offset text slightly to the left
                    textcoords='offset points',
                    color=color,
                    fontweight='bold',
                    fontsize=8)

        if len(svd.explained_variance_ratio_) < n_components:
            print(f"Layer {layer} has less than {n_components} components. Only {len(svd.explained_variance_ratio_)} components are present.")
            n_components = len(svd.explained_variance_ratio_)

    plt.xlabel('Number of Components', fontsize=20)
    plt.ylabel('Cumulative Explained Variance', fontsize=20)
    plt.title(f'Explained Variance by Component for each Layer in {model_type}', fontsize=20)
    plt.grid(True)
    
    # Use a custom legend that shows only a few representative layers to avoid overcrowding
    legend_layers = [0, num_layers//4, num_layers//2, 3*num_layers//4, num_layers-1]
    legend_handles = [plt.Line2D([0], [0], color=colormap(i/(num_layers-1)), label=f'Layer {i}') 
                     for i in legend_layers]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'regression/{model_type}_explained_variance_by_component_{n_components}.png', dpi=300)
    # plt.show()


    # Now save the reduced embeddings for each layer in a dictionary in the same format as that of the original embeddings.
    embeds_dict = {} # This creates a dictionary with the same keys as the original embeddings.
    for k, v in labels_mapping.items():
        embeds_dict[v] = []
    for layer in range(num_layers):
        layer_labels = get_transformer_embeddings_from_dict(labels, embeds_dict=embeds, layer=layer) # Preserves the return order as that of 'labels'.
        svd = TruncatedSVD(n_components=n_components)
        layer_labels_svd = svd.fit_transform(layer_labels)

        for i, label in labels_mapping.items():
            embeds_dict[label].append(layer_labels_svd[i])


    
    # Save the dictionary.
    with open(os.path.join(STORE_PATH_MAPPING[model_type], f'embeds_with_label_dict_reduced_svd_n_components_{n_components}.pkl'), 'wb') as f:
        pickle.dump(embeds_dict, f)




