import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image


def get_distribution(path: str) -> Dict[int, str]:
    classes = os.listdir(path)
    d = {}
    for c in classes:
        d[c] = len(os.listdir(os.path.join(path, c)))
    return d


def plot_distibution(
    d: Dict[str,int], 
    title: str = 'training set',
    figsize: Tuple[int, int] = (12, 5),
):
    plt.subplots(figsize=figsize)

    d_list = sorted(list(d.items()), key=lambda x: x[0])
    classes = [x[0] for x in d_list]
    frequencies = [x[1] for x in d_list]
    
    plt.bar(classes, frequencies)
    plt.xlabel('classes')
    plt.ylabel('frequencies')
    plt.title(title)


def load_image(path_to_image: str):
    return np.array(image.load_img(path_to_image))


def load_random_image(path_to_class):
    images_path = os.listdir(path_to_class)
    return load_image(os.path.join(path_to_class, np.random.choice(images_path)))


def move_object(src, dest):
    filename = os.path.basename(src)
    if not os.path.exists(dest):
        os.makedirs(dest, exist_ok=True)
    os.rename(src, os.path.join(dest, filename))


def get_number_of_steps(n_samples: int, batch_size: int) -> int:
    return int(n_samples / batch_size)


def get_model1(input_size, num_class, with_batch_normalization: bool = False):
    i = L.Input(shape=input_size)
    x = L.Conv2D(32, (3, 3), activation='relu')(i)
    if with_batch_normalization:
        x = L.BatchNormalization()(x)
    x = L.MaxPooling2D((2, 2))(x)
    x = L.Conv2D(64, (3, 3), activation='relu')(x)
    if with_batch_normalization:
        x = L.BatchNormalization()(x)
    x = L.MaxPooling2D((2, 2))(x)
    x = L.Conv2D(128, (3, 3), activation='relu')(x)
    if with_batch_normalization:
        x = L.BatchNormalization()(x)
    x = L.MaxPooling2D((2, 2))(x)
    x = L.Conv2D(128, (3, 3), activation='relu')(x)
    if with_batch_normalization:
        x = L.BatchNormalization()(x)
    x = L.MaxPooling2D((2, 2))(x)
    x = L.Flatten()(x)
    x = L.Dense(512, activation='relu')(x)
    o = L.Dense(num_class, activation='softmax')(x)
    return Model(i, o)


def get_model2(input_size, num_class):
    vgg = tf.keras.applications.VGG16(
        input_shape=input_size, 
        include_top=False, 
        weights='imagenet',
    )
    x = L.Flatten()(vgg.output)
    x = L.Dense(num_class, activation='softmax')(x)

    vgg.trainable = False

    return Model(vgg.input, x)


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def get_labels(filenames):
    labels = []
    for filename in filenames:
        label = filename.split('/')[0]
        if label not in labels:
            labels.append(label)
    return labels
