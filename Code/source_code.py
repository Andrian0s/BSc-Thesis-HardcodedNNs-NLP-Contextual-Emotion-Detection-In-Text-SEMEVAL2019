import numpy as np
import math
import re
import io
import emoji
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

EMBEDDING_DIM = 100

TOTAL_CLASSES = 4                    # Number of classes - Happy, Sad, Angry, Others
NUM_BATCHES = 100                  # The batch size to be chosen for training the model.
LABEL2EMOTION = {0: "others", 1: "happy", 2: "sad", 3: "angry"}
EMOTION2LABEL = {"others": 0, "happy": 1, "sad": 2, "angry": 3}
IGNORE_LIST = ['a', "the", "is"]  # The ignore_list to be used with soft_clean_text
ALL_LETTERS = 'abcdefghijklmnopqrstuvwxyz'  # All letters in a single string
FEATURES = {"Bag_of_Words": "bow.npy", "word2vec": "w2v_embeddings.npy", "TF_IDF": "tf_idf.npy"}  # features to in/out
DIRECTORIES = {"features": "features/", "weights": "weights/"}
TRAIN_DATA_PATH = "../Data/train.txt"
DEV_DATA_PATH = "../Data/devwithoutlabels.txt"
OUTPUT_BOW_AND_TRAIN = "../Data/train_and_bow.txt"
OUTPUT_BOW_AND_TEST = "../Data/test_and_bow.txt"
UNKNOWN_WORD = "word_not_found"
EMOTICONS_START = [":", ";", "<", "="]
MAX_BOW_VALUE = 4
MAX_SENTENCE_LEN = 40
MINIMUM_IMPROVEMENT = 0.001
np.set_printoptions(suppress=True)


all_words_dict = dict()
word_to_index = dict()
inverse_document_frequency = dict()
saved_pre = dict()
validation_errors = []
classification_results = []
embedding_model = ""
embedding_words = []
run_counter = 0


def input_data(feature_type, run_name, existing, sample, overwrite, two_class, hierarchical, initial):
    """Load the neural network features, either from an existing save or generates them.
            Input:
                feature_type : What neural network features we will use. (Bag of Words, Word2vec).
                existing : A boolean showing whether to use existing from a file or not.
                sample : A boolean that shows whether we want our data to be sampled or not.
                overwrite: A boolean that shows if we generated the features, whether to save them for future use.
                two_class: Whether we are inputting data for a two_class classifier
                hierarchical: Whether this is for the hierarchical classifier.
                initial: Whether this is the initial run, where we only save the test set.
            Output:
                train_feat : The neural network features for the training data.
                train_labels : The neural network labels for the training data.
                val_feat : The neural network features for the validation data.
                val_labels : The neural network labels for the validation data.
                test_feat : The neural network features for the test data.
                test_labels : The neural network labels for the test data.
    """

    data_type = FEATURES[feature_type]
    if initial:
        n_classes = TOTAL_CLASSES
        train_conversations, train_labels, test_conversations, test_labels = pre_process_data(TRAIN_DATA_PATH, "train")
        if hierarchical and not two_class:  # This means we need to convert our data set into 0-3 ignoring Others Class.
            indices_of_new_labels = [index for index, elem in enumerate(test_labels) if elem > 0]
            n_classes = 3
            print(len(indices_of_new_labels))
            test_labels = [i - 1 for i in test_labels if i > 0]
            full_test = np.array(test_conversations)
            min_test = full_test[indices_of_new_labels]
            test_conversations = min_test
            test_labels = test_labels
        if two_class:
            n_classes = 2
            test_labels = [min(i, 1) for i in test_labels]
        populate_all_words_dict(train_conversations)
        populate_word_to_index_dictionary()
        test_conversations = remove_unknown_words(test_conversations, replace=False)
        test_labels = change_labels_into_neural_network_friendly(test_labels, n_classes)
        if feature_type == "word2vec":
            if embedding_model == "":
                create_embedding_model(train_conversations)
            test_feat = create_w2v_embeddings_for_data_set(test_conversations)
        else:  # Bag of Words, TF_IDF
            test_feat = create_list_of_bag_of_words(test_conversations)

            if feature_type == "Bag_of_Words":
                test_feat = bags_of_words_to_bow_neural_network_feature(test_feat)
            if feature_type == "TF_IDF":
                generate_inverse_document_frequency(len(train_conversations))
                test_feat = bags_of_words_to_tf_idf_neural_network_feature(test_feat)
        if overwrite:
            np.save((DIRECTORIES["features"] + "test_" + data_type), test_feat)
            np.save((DIRECTORIES["features"] + "test_labels_" + data_type), test_labels)
        return test_feat, test_labels

    sample_f_name = ""
    number_of_classes = 2 if two_class else 4
    if sample:
        sample_f_name = "sample_"

    if existing:
        return np.load(DIRECTORIES["features"] + sample_f_name + "train_" + data_type), np.load(DIRECTORIES["features"] +
                    sample_f_name + "train_labels_" + data_type), np.load(DIRECTORIES["features"] + sample_f_name +
                    "val_" + data_type), np.load(DIRECTORIES["features"] + sample_f_name + "val_labels_" + data_type), \
                    np.load(DIRECTORIES["features"] + sample_f_name + "test_" + data_type), \
                    np.load(DIRECTORIES["features"] + sample_f_name + "test_labels_" + data_type)
    else:
        print("Processing training and test data...")
        train_conversations, train_labels, _, _ = pre_process_data(TRAIN_DATA_PATH, "train")
        # dev_indices, dev_conversations = pre_process_data(TEST_DATA_PATH, "test")
        if hierarchical and not two_class:  # This means we need to convert our data set into 0-3 ignoring Others Class.
            number_of_classes = 3
            indices_of_new_labels = [index for index, elem in enumerate(train_labels) if elem > 0]
            print(len(indices_of_new_labels))
            train_labels = [i-1 for i in train_labels if i > 0]
            full_train = np.array(train_conversations)
            min_train = full_train[indices_of_new_labels]
            train_conversations = min_train
        if two_class:
            train_labels = [min(i, 1) for i in train_labels]

        print("Splitting into training and validation set")
        if sample:
            train_conversations, train_labels = get_equal_class_sample(train_conversations, train_labels, number_of_classes)
        else:
            train_conversations = np.array(train_conversations)
            train_labels = np.array(train_labels)
        train_conversations, train_labels, val_conversations, val_labels = \
            extract_test_or_val_set(train_conversations, train_labels, number_of_classes)

        print("Generating neural network features...")

        val_conversations = remove_unknown_words(val_conversations, replace=False)

        train_labels = change_labels_into_neural_network_friendly(train_labels, number_of_classes)
        val_labels = change_labels_into_neural_network_friendly(val_labels, number_of_classes)
        print("Extracting tokens...")

        if feature_type == "word2vec":
            train_feat = create_w2v_embeddings_for_data_set(train_conversations)
            val_feat = create_w2v_embeddings_for_data_set(val_conversations)

        else:  # Bag of Words, TF_IDF
            train_feat = create_list_of_bag_of_words(train_conversations)
            val_feat = create_list_of_bag_of_words(val_conversations)
            if feature_type == "Bag_of_Words":
                train_feat = bags_of_words_to_bow_neural_network_feature(train_feat)
                val_feat = bags_of_words_to_bow_neural_network_feature(val_feat)

            if feature_type == "TF_IDF":
                train_feat = bags_of_words_to_tf_idf_neural_network_feature(train_feat)
                val_feat = bags_of_words_to_tf_idf_neural_network_feature(val_feat)

        if overwrite:
            np.save((DIRECTORIES["features"] + sample_f_name + "train_" + data_type), train_feat)
            np.save((DIRECTORIES["features"] + sample_f_name + "val_" + data_type), val_feat)
            np.save((DIRECTORIES["features"] + sample_f_name + "train_labels_" + data_type), train_labels)
            np.save((DIRECTORIES["features"] + sample_f_name + "val_" + data_type), val_labels)

        return train_feat, train_labels, val_feat, val_labels


def pre_process_unseen(feature_type, test_conversations):
    """ Pre_process unseen test data so that it can be classified with previous weights.
    """

    test_conversations = remove_unknown_words(test_conversations, replace=False)
    if feature_type == "word2vec":
        test_feat = create_w2v_embeddings_for_data_set(test_conversations)
    else:  # Bag of Words, TF_IDF
        test_feat = create_list_of_bag_of_words(test_conversations)
        if feature_type == "Bag_of_Words":
            test_feat = bags_of_words_to_bow_neural_network_feature(test_feat)
        if feature_type == "TF_IDF":
            test_feat = bags_of_words_to_tf_idf_neural_network_feature(test_feat)
    return test_feat


def pre_process_data(data_file_path, mode):
    """ Load data from a file, process and return indices, conversations and labels in separate lists.
        If already calculated, return them instead.
            Input:
                data_file_path : Path to train/test file to be processed.
                mode : "train" mode returns labels. "test" mode doesn't return labels.
            Output:
                indices : Unique conversation ID list.
                conversations : List of 3 turn conversations, processed and each turn separated (removed <eos> tag for now).
                labels : [Only available in "train" mode] List of labels.
    """

    global saved_pre
    if mode == "test":
        conversations = []
        with io.open(data_file_path, encoding="utf8") as f_input:
            f_input.readline()
            for (index, line) in enumerate(f_input):
                # Convert multiple instances of . ? ! , to single instance
                # okay...sure -> okay . sure
                # okay???sure -> okay ? sure
                # Add whitespace around such punctuation
                # okay!sure -> okay ! sure
                repeated_chars = ['.', '?', '!', ',']
                for c in repeated_chars:
                    line_split = line.split(c)
                    while True:
                        try:
                            line_split.remove('')
                        except:
                            break
                    c_space = ' ' + c + ' '
                    line = c_space.join(line_split)

                line = line.strip().split('\t')
                for i in range(1, 4):
                    line[i] = soft_clean_text(line[i])
                    line[i] = line[i].split(" ")
                    line[i] = ' '.join(line[i])

                # conv = ' <eos> '.join(line[1:4])
                conv = ' '.join(line[1:4])
                # Remove any duplicate spaces
                duplicates_space_pattern = re.compile(r'\ +')
                conv = re.sub(duplicates_space_pattern, ' ', conv).split()[0:min(len(conv), MAX_SENTENCE_LEN)]
                conversations.append(conv)
            test_conversations = conversations
            return test_conversations

    conversations = []
    labels = []
    with io.open(data_file_path, encoding="utf8") as f_input:
        f_input.readline()
        for (index, line) in enumerate(f_input):
            # Convert multiple instances of . ? ! , to single instance
            # okay...sure -> okay . sure
            # okay???sure -> okay ? sure
            # Add whitespace around such punctuation
            # okay!sure -> okay ! sure
            repeated_chars = ['.', '?', '!', ',']
            for c in repeated_chars:
                line_split = line.split(c)
                while True:
                    try:
                        line_split.remove('')
                    except:
                        break
                c_space = ' ' + c + ' '
                line = c_space.join(line_split)

            line = line.strip().split('\t')
            for i in range(1, 4):
                line[i] = soft_clean_text(line[i])
                line[i] = line[i].split(" ")
                line[i] = ' '.join(line[i])

            # conv = ' <eos> '.join(line[1:4])
            conv = ' '.join(line[1:4])
            # Remove any duplicate spaces
            duplicates_space_pattern = re.compile(r'\ +')
            conv = re.sub(duplicates_space_pattern, ' ', conv).split()[0:min(len(conv), MAX_SENTENCE_LEN)]

            if mode == "train":
                label = EMOTION2LABEL[line[4]]
                conversations.append(conv)
                labels.append(label)

            else:
                conversations.append(conv)
    train_conversations, train_labels, test_conversations, test_labels = extract_test_or_val_set(conversations, labels)
    saved_pre["train_conversations"] = train_conversations
    saved_pre["train_labels"] = train_labels
    saved_pre["test_conversations"] = test_conversations
    saved_pre["test_labels"] = test_labels

    return train_conversations, train_labels, test_conversations, test_labels


def soft_clean_text(single_text):
    """ Splits the single_text into a single string of  lowercase and stop_list removal, emojis and emoticons spaced.
            Input:
                single_text : A single sentence (in this case, 3 concatenated).
            Output:
                cleaned_text : The text tokenized, lowercase and stop-list_removed.
                """

    for emoticon_starter in EMOTICONS_START:  # {":", ";", "<", "="}
        single_text = single_text.replace(emoticon_starter, " "+emoticon_starter)  # space before emoticon starter
    index = 0
    while index < len(single_text):  # putting a space before emojis so they are treated as a  seperate word
        if single_text[index] in emoji.UNICODE_EMOJI:
            single_text = single_text[:index] + ' ' + single_text[index:]
            index += 1
        index += 1
    words = single_text.split(" ")
    cleaned_text = " ".join([w.lower() for w in words if w not in IGNORE_LIST])
    return cleaned_text


def extract_test_or_val_set(conversations, labels, number_of_classes=4):
    """ Splits the training data_set into train and val or test set keeping the equal class distribution.
            Input:
                conversations: The data's conversations
                labels: The data's labels
                number_of_classes: The number of classes we split for
            Output:
                train_data: The training conversations
                train_labels: The training labels
                test_data: The test/val conversations
                test_labels: The test/val labels
   """

    train_labels = []
    test_labels = []
    conversations = np.array(conversations)
    for i in range(number_of_classes):
        indexes = [index for index, elem in enumerate(labels) if elem == i]
        eighty_percent = int(len(indexes) / 5 * 4)
        train_labels.extend(indexes[:eighty_percent])
        test_labels.extend(indexes[eighty_percent:])
    train_labels = np.array(train_labels).flatten()
    # np.random.shuffle(train_labels) no need, gets shuffled at each epoch
    test_labels = np.array(test_labels).flatten()
    # np.random.shuffle(test_labels) no need, doesn't matter
    train_data = conversations[train_labels]
    test_data = conversations[test_labels]
    train_labels = np.take(labels, train_labels)
    test_labels = np.take(labels, test_labels)
    return train_data, train_labels, test_data, test_labels


def populate_all_words_dict(data_set):
    """ Populates the dictionary of all words counting the indexes of each occurence
            Input:
                data_set: The data we are adding to our dictionary.
    """

    global all_words_dict
    for index, sentence in enumerate(data_set):
        for word in sentence:
            if not(word in all_words_dict):
                all_words_dict[word] = set()
            all_words_dict[word].add(index)


def generate_inverse_document_frequency(total_train_samples):
    """ Populates the inverse document frequency dictionary.
            Input:
                data_set: The total number of train samples
    """

    global inverse_document_frequency
    max_idf = 0
    for word in all_words_dict.keys():
        inverse_document_frequency[word] = math.log10(total_train_samples / len(all_words_dict[word]))
        max_idf = max(max_idf, inverse_document_frequency[word])
    print("Max idf is ")
    print(max_idf)


def count_minimum_class(full_labels, number_of_classes):
    """ Given a list of labels, finds the cardinality of the class with least labels.
            Input:
                full_labels : The labels of the data set
            Output:
                The cardinality of the class with the least occurences.
    """

    full_labels = np.array(full_labels)
    minimum_class_cardinality = 20000000
    for i in range(number_of_classes):
        samples = np.count_nonzero(full_labels == i)
        minimum_class_cardinality = min(samples, minimum_class_cardinality)
    return minimum_class_cardinality


def create_embedding_model(train_conv):
    """ Given the conversations of the training set, creates a word embedder.
            Input:
                train_conv : conversations of the training set
            Output:
                Generates the word embedder of word2vec and the words of the model stored in a global variable.
    """
    global embedding_model
    global embedding_words
    sentences = []
    for sentence in train_conv:
        sentences.append(sentence)
    embedding_model = Word2Vec(sentences, min_count=3, sg=1, size=EMBEDDING_DIM)
    embedding_words = embedding_model.wv


def get_equal_class_sample(full_train, full_labels, number_of_classes):
    """ Given the training data and labels, returns the data sampled with equally distributed classes
            Input:
                full_train: The data's conversations
                full_labels: The data's labels
            Output:
                sample_train: The new sample train data
                sample_labels: The new sample labels
    """

    class_minimum = count_minimum_class(full_labels, number_of_classes)
    # print(class_minimum)
    sample_labels = []
    for i in range(number_of_classes):
        temp_indexes = [index for index, elem in enumerate(full_labels) if elem == i]
        sample_labels.extend(temp_indexes[:class_minimum])
    sample_labels = np.array(sample_labels)
    full_train = np.array(full_train)
    sample_train = full_train[sample_labels]
    sample_labels = np.take(full_labels, sample_labels)

    return sample_train, sample_labels.tolist()


def split_data_set_balanced(data, data_labels, number_of_classes):
    """ Splits the data_set into train, val and test set keeping the equal class distribution.
            Input:
                data: The data's conversations
                data_labels: The data's labels
            Output:
                train_data: The training conversations
                train_labels: The training labels
                val_data: The validation conversations
                val_labels: The validation labels
                test_data: The test conversations
                test_labels: The test labels
    """

    print(number_of_classes)
    train_labels = []
    val_labels = []
    test_labels = []
    for i in range(number_of_classes):
        indexes = [index for index, elem in enumerate(data_labels) if elem == i]
        sixty_percent = int(len(indexes) / 5 * 3)
        eighty_percent = int(len(indexes) / 5 * 4)
        train_labels.extend(indexes[:sixty_percent])
        val_labels.extend(indexes[sixty_percent:eighty_percent])
        test_labels.extend(indexes[eighty_percent:])
    train_labels = np.array(train_labels).flatten()
    np.random.shuffle(train_labels)
    val_labels = np.array(val_labels).flatten()
    np.random.shuffle(val_labels)
    test_labels = np.array(test_labels).flatten()
    np.random.shuffle(test_labels)
    train_data = data[train_labels]
    val_data = data[val_labels]
    test_data = data[test_labels]
    train_labels = np.take(data_labels, train_labels)
    val_labels = np.take(data_labels, val_labels)
    test_labels = np.take(data_labels, test_labels)
    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def create_list_of_bag_of_words(data_set):
    """ Makes the data set into Bag of Words representation. Limits to 5 of each
            Input:
                data_set : The data we would like to make to Bag of Words.
                mode : Whether this is part of the training or test data.
            Output:
                data_bow : A dictionary of index to Bag of Words representation.
    """

    data_bow = []
    for sentence in data_set:
        current_bow = {}
        for word in sentence:
            if word in current_bow:
                current_bow[word] = min(current_bow[word]+1, MAX_BOW_VALUE)
            else:
                current_bow[word] = 1
        data_bow.append(current_bow)
    return data_bow


def remove_unknown_words(test_data, replace):
    """ Removes unknown words that cannot be corrected. Replaces if replace is true
            Input:
                test_data : The test data we would like to remove the unknown words from.
                replace : Whether we want to replace the word with the correction or not.
            Output:
                the test data with the unknown words either removed or replaced.
    """

    for (j, sentence) in enumerate(test_data):
        for (i, word) in enumerate(sentence):
            if not (word in all_words_dict):
                new_word = correction(word)
                if new_word == UNKNOWN_WORD:
                    test_data[j][i] = ""
                else:
                    if replace:
                        test_data[j][i] = new_word
                    else:
                        test_data[j][i] = ""
    return [list(filter(None, sentence)) for sentence in test_data]


def compare_data_set_changes(text, bow, output_file):
    """ Prints the original data set and the Bow representation to a file.
            Input:
                text : The original list of texts.
                bow: A dictionary of index to Bag of Words representation.
                output_file : The file we want to output at.
    """

    with io.open(output_file, "w", encoding="utf8") as f_out:
        for i in range(len(text)):
            text[i] = " ".join(text[i])
            f_out.write(str(i) + " : ")
            f_out.write(text[i])
            f_out.write("  ")
            f_out.write(str(bow[i]))
            f_out.write("\n")


def correction(word):
    """ Most probable spelling correction for word.
            Input:
                word: The word we are looking to correct.
            Output:
                the word correct or the word itself.
    """

    maxi = 0
    max_word = "word_not_found"
    for key in known(edits1(word)):
        maxi = max(len(all_words_dict[key]), maxi)
        if maxi == len(all_words_dict[key]):
            max_word = key
    return max_word


def known(words):
    """ The subset of `words` that appear in the dictionary of all_words_dict.
            Input:
                words: the edited words.
            Output:
                the set of words in all_words_dict.
    """

    return set(w for w in words if w in all_words_dict)


def edits1(word):
    """ All edits that are one edit away from `word`.
            Input:
                words: the word to find edits of
            Output:
                the set of edits of that word.
    """

    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in ALL_LETTERS]
    inserts = [L + c + R for L, R in splits for c in ALL_LETTERS]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    """ All edits that are two edits away from `word`. """

    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


def populate_word_to_index_dictionary():
    """ Populates the word to index dictionary to be used for nn-features. """

    global word_to_index
    dict_keys = sorted(all_words_dict.keys())
    for (index, key) in enumerate(dict_keys):
        word_to_index[key] = index


def bags_of_words_to_tf_idf_neural_network_feature(data_set):
    """ Transforms the data set into the corresponding nn-friendly np array as TF_IDF """

    bow_features = np.empty((len(all_words_dict), len(data_set)), float)
    for index, bow in enumerate(data_set):
        bow_features[:, index] = bow_to_neural_network_tf_idf(bow)
    return bow_features


def bags_of_words_to_bow_neural_network_feature(data_set):
    """ Transforms the data set into the corresponding nn-friendly np array as BoW. """

    bow_features = np.empty((len(all_words_dict), len(data_set)), float)
    for index, bow in enumerate(data_set):
        bow_features[:, index] = bow_to_neural_network_bow(bow)
    return bow_features


def bow_to_neural_network_tf_idf(bag_of_words):
    """ Uses the bow and word to index dictionary to create nn friendly features. """

    nn_feature = np.zeros(len(all_words_dict), float)
    total_idf = sum(bag_of_words.values())
    for key in bag_of_words.keys():
        nn_feature[word_to_index[key]] = bag_of_words[key] / total_idf * inverse_document_frequency[key]
    return nn_feature


def bow_to_neural_network_bow(bag_of_words):
    """ Uses the bow and word to index dictionary to create nn friendly features. """

    nn_feature = np.zeros(len(all_words_dict), float)
    for key in bag_of_words.keys():
        nn_feature[word_to_index[key]] = bag_of_words[key]
    return nn_feature


def create_w2v_embedding_for_sentence(sentence):
    """ Creates the embeddings for the sentence.
            Input:
                embedding_model : The model that creates the word embeddings.
                sentence: the sentence to create the embedding.
            Output:
                the embedding of the sentence (4000 floats, 0s for less than 40 words).
    """

    sentence_embedding = np.zeros(MAX_SENTENCE_LEN * EMBEDDING_DIM, float)
    sentence = [word for word in sentence if word in embedding_words]
    if len(sentence) > 0:
        sentence_embedding[:(len(sentence) * EMBEDDING_DIM)] = np.reshape(embedding_model.wv[sentence],
                                                                      EMBEDDING_DIM * len(sentence))
    return sentence_embedding


def create_w2v_embeddings_for_data_set(data_set):
    """ Creates the embeddings of the data_set based on the embedding_model.
            Input:
                data_set : The data_set we want to create the embeddings for
            Output:
                the corresponding embeddings for the data-set.
    """

    embeddings = np.empty(((MAX_SENTENCE_LEN * EMBEDDING_DIM), len(data_set)), float)
    for i in range(len(data_set)):
        embeddings[:, i] = create_w2v_embedding_for_sentence(data_set[i])
    return embeddings


def change_labels_into_neural_network_friendly(labels, number_of_classes):
    """ Changes the shape of labels into a neural network friendly form. ex: (0,0,1,0)
            Input:
                labels : the labels of the data_set.
            Output:
                the labels in a n-n friendly-form.
    """

    y = np.zeros((len(labels), number_of_classes))
    for i in range(0, len(labels)):
        y[i, labels[i]] = 1
    return y


def train_ff_backprop_nn_dynamic(eta, n_epoch, train_features, train_labels, val_features, val_labels, num_of_classes,
                                 activation="relu", neurons_list=[50, 50]):
    """ Train the neural network using the training and validation data (dynamic amount of hidden layers) 0 for Log_Reg.
            Input:
                eta: The learning rate of the neural network.
                n_epoch: The number of epochs (repetitions).
                train_features: the features of the training data.
                train_labels: the labels for the training data.
                val_features: the features of the validation data.
                val_labels: the labels of the validation data.
                num_of_classes: the number of classes in our set.
                activation: The activation function to use "sigmoid" or "relu".
                neurons_list: The list of neurons. [] for Logistic Regressions
            Output:
                The weights and bias weights lists(results of training).
    """

    # define the size of each of the layers in the network
    print(train_features.shape)
    print(train_labels.shape)
    val_errors = []
    input_size, n_samples = train_features.shape
    n_input_layer = input_size
    n_output_layer = num_of_classes

    batch_size = math.ceil(n_samples / NUM_BATCHES)

    weights = []
    bias_weights = []
    prev_layer = n_input_layer

    prev_weights = []
    prev_biases = []
    prev_error = 10000000

    if activation == "relu":  # He Init
        numerator = 2
    else:  # Xavier_Init for Sigmoid
        numerator = 1

    for neurons in neurons_list:  # Initialise weights and biases
        weights.append(np.random.randn(neurons, prev_layer) * np.sqrt((numerator / prev_layer)))
        if activation == "sigmoid":
            bias_weights.append(np.zeros((neurons,)))
        else:
            bias_weights.append(np.random.randn(neurons)/100+0.01)
        prev_layer = neurons

    # Initialise the last layer
    if len(neurons_list) == 0:  # Logistic Regression Case
        weights.append(np.random.randn(n_output_layer, n_input_layer) * np.sqrt((1 / n_input_layer)))
    else:  # 1 or more Hidden Layers
        weights.append(
            np.random.randn(n_output_layer, neurons_list[-1]) * np.sqrt((1 / neurons_list[-1])))  # Xavier even in relu.

    bias_weights.append(np.zeros(n_output_layer))

    # Train the network
    errors = np.zeros((n_epoch,))
    for i in range(0, n_epoch):
        # print(i)
        # We will shuffle the order of the samples each epoch
        shuffled_idxs = np.random.permutation(n_samples)
        confusion_matrix = np.zeros((num_of_classes, num_of_classes), int)
        for batch in range(0, NUM_BATCHES):
            # Initialise the gradients for each batch
            dif_weights = []
            dif_bias_weights = []
            for index in range(len(weights)):  # initialising the weigh ts before each batch
                dif_weights.append(np.zeros(weights[index].shape))
                dif_bias_weights.append(np.zeros(bias_weights[index].shape))
            # Loop over all the samples in the batch
            for j in range(0, batch_size):
                # Input (random element from the dataset)
                if batch * batch_size + j >= len(shuffled_idxs) - 1:
                    break
                indx = shuffled_idxs[batch * batch_size + j]
                x = train_features[:, indx]
                acts = []
                outs = []
                out_delta = []
                desired_output = train_labels[indx]
                prev_out = x
                #  Feed forward
                for index in range(len(weights)):
                    # Neural activation
                    acts.append(np.dot(weights[index], prev_out) + bias_weights[index])
                    if index == len(weights) - 1:  # Softmax for Last layer
                        exps = np.exp(acts[index] - np.max(acts[index], axis=0, keepdims=True))
                        outs.append(exps / np.sum(exps, axis=0, keepdims=True))
                    elif activation == "sigmoid":  # Activation with Sigmoid
                        outs.append(1 / (1 + np.exp(-acts[index])))
                    else:  # Activation with Relu
                        outs.append(acts[index] * (acts[index] > 0))
                    prev_out = outs[index]
                # Compute the error signal
                e_n = np.sum(desired_output * np.log(outs[-1]))

                # Backpropagation for Softmax layer
                out_delta.append(desired_output - outs[-1])
                if len(neurons_list) == 0:  # Logistic Regression Case
                    dif_weights[len(dif_weights) - 1] += np.outer(out_delta[0], x)
                    dif_bias_weights[len(dif_bias_weights) - 1] += out_delta[0]
                else:
                    dif_weights[len(dif_weights) - 1] += np.outer(out_delta[0], outs[-2])
                    dif_bias_weights[len(dif_bias_weights) - 1] += out_delta[0]

                # Backpropagation for the rest of the layers
                for index in range(len(outs) - 2, -1, -1):
                    #  Backpropagation: Current Layer to the previous layer
                    if activation == "sigmoid":
                        out_delta.append(outs[index] * (1 - outs[index]) * np.dot(weights[index + 1].T, out_delta[-1]))

                    if activation == "relu":
                        out_delta.append(np.heaviside(outs[index], 0) * np.dot(weights[index + 1].T, out_delta[-1]))

                    if index == 0:  # This is the last layer, so instead of output we use the input feat.
                        dif_weights[index] += np.outer(out_delta[-1], x)
                    else:
                        dif_weights[index] += np.outer(out_delta[-1], outs[index - 1])
                    dif_bias_weights[index] += out_delta[-1]

                correct_value = np.argmax(desired_output)
                predicted_value = np.argmax(outs[-1])

                confusion_matrix[int(correct_value), int(predicted_value)] += 1
                # Store the error per epoch
                errors[i] = errors[i] + (-1 / n_samples) * e_n

            # After each batch update the weights using accumulated gradients
            for index in range(len(weights)):
                weights[index] += eta * dif_weights[index] / batch_size
                bias_weights[index] += eta * dif_bias_weights[index] / batch_size

        # print(" Training set Results : Epoch ", i + 1, ": error = ", errors[i])
        # print(confusion_matrix)
        print("Validation set Results :")
        conf, error = classify_with_ff_backprop_nn_dynamic(val_features, val_labels, weights, bias_weights,
            num_of_classes, activation)
        val_errors.append(error)
        if i % 5 == 0:
            if (error+MINIMUM_IMPROVEMENT) < prev_error:
                print("PREV ERROR IS: ")
                print(prev_error)
                prev_error = error
                prev_weights = weights
                prev_biases = bias_weights
            else:
                validation_errors.append(val_errors)
                return prev_weights, prev_biases
        print(error)
    validation_errors.append(val_errors)
    return weights, bias_weights


def classify_with_ff_backprop_nn_dynamic(test_features, test_labels, weights, bias_weights, num_of_classes, activation):
    """  Classifies the data using the weights and bias weights trained.
            Input:
                test_features the test data we want to classify.
                test_labels: the correct labels for the test data.
                weights: the weights of the trained neural network.
                bias_weights: the weights of the trained neural network.
                num_of_classes: the number of classes we are classifying for.
                activation: The activation function to use "sigmoid" or "relu".
            Output:
                prints to the console the macro_f1, confusion_matrix and accuracy.
                returns the cross entropy loss.
    """

    print(test_features.shape)
    n = test_features.shape[1]
    correct_value = np.zeros((n,))
    predicted_value = np.zeros((n,))
    confusion_matrix = np.zeros((num_of_classes, num_of_classes), int)
    error = 0
    for i in range(0, n):
        x = test_features[:, i]
        desired_output = test_labels[i]
        correct_value[i] = np.argmax(desired_output)
        acts = []
        outs = []
        prev_out = x
        # Feed forward Neural Network
        for index in range(len(weights)):
            # Neural activation
            acts.append(np.dot(weights[index], prev_out) + bias_weights[index])
            if index == len(weights) - 1:  # Softmax for Last Layer
                exps = np.exp(acts[index] - np.max(acts[index], axis=0, keepdims=True))
                outs.append(exps / np.sum(exps, axis=0, keepdims=True))
            elif activation == "sigmoid":  # activate with Sigmoid
                outs.append(1 / (1 + np.exp(-acts[index])))
            else:  # activate with Relu
                outs.append(acts[index] * (acts[index] > 0))
            prev_out = outs[index]
        # Compute the error signal
        e_n = np.sum(desired_output * np.log(outs[-1]))

        predicted_value[i] = np.argmax(outs[-1])
        # Count the total number of correct classifications
        confusion_matrix[int(correct_value[i]), int(predicted_value[i])] += 1
        error += (-1 / n) * e_n

    print(np.sum(np.diagonal(confusion_matrix)))
    print("Macro F1 " + str(macro_f1(confusion_matrix)))
    print("Micro F1 " + str(micro_f1(confusion_matrix)))
    print(confusion_matrix)
    return confusion_matrix, error


def classify_with_ff_backprop_nn_dynamic_hierarchical(test_features_first_layer, test_features_second_layer,
                                                      test_labels, weights_list, bias_weights_list, activation_list):
    """  Classifies the data using the weights and bias weights trained through our 2 classifiers.
            Input:
                test_features_first_layer: the test data we want to classify in the feature form of the first layer.
                test_features_second_layer: the test data we want to classify in the feature form of the first layer.
                test_labels: the correct labels for the test data.
                weights_list: The weights of our 2 trained neural networks.
                bias_weights_list: The bias of our 2 trained neural networks.
                activation_list: The activation function to use on each classifier.
            Output:
                prints to the console the macro_f1, confusion_matrix and accuracy.
                returns the cross entropy loss.
    """

    print(test_features_first_layer.shape)
    n = test_features_first_layer.shape[1]
    confusion_matrix = np.zeros((TOTAL_CLASSES, TOTAL_CLASSES), int)
    error = 0
    unclassified_indexes = []

    for i in range(0, n):
        x = test_features_first_layer[:, i]
        desired_output = test_labels[i]
        correct_value = np.argmax(desired_output)
        acts = []
        outs = []
        prev_out = x
        # Feed forward Neural Network
        for index in range(len(weights_list[0])):
            # Neural activation
            acts.append(np.dot(weights_list[0][index], prev_out) + bias_weights_list[0][index])
            if index == len(weights_list[0]) - 1:  # Softmax for Last Layer
                exps = np.exp(acts[index] - np.max(acts[index], axis=0, keepdims=True))
                outs.append(exps / np.sum(exps, axis=0, keepdims=True))
            elif activation_list[0] == "sigmoid":  # activate with Sigmoid
                outs.append(1 / (1 + np.exp(-acts[index])))
            else:  # activate with Relu
                outs.append(acts[index] * (acts[index] > 0))
            prev_out = outs[index]
        # Compute the error signal
        predicted_value = np.argmax(outs[-1])
        # Count the total number of correct classifications
        if predicted_value == 0:
            confusion_matrix[int(correct_value), int(predicted_value)] += 1
        else:
            unclassified_indexes.append(i)
    print(confusion_matrix)
    for i in unclassified_indexes:
        x = test_features_second_layer[:, i]
        desired_output = test_labels[i]
        correct_value = np.argmax(desired_output)
        acts = []
        outs = []
        prev_out = x
        # Feed forward Neural Network
        for index in range(len(weights_list[1])):
            # Neural activation
            acts.append(np.dot(weights_list[1][index], prev_out) + bias_weights_list[1][index])
            if index == len(weights_list[1]) - 1:  # Softmax for Last Layer
                exps = np.exp(acts[index] - np.max(acts[index], axis=0, keepdims=True))
                outs.append(exps / np.sum(exps, axis=0, keepdims=True))
            elif activation_list[1] == "sigmoid":  # activate with Sigmoid
                outs.append(1 / (1 + np.exp(-acts[index])))
            else:  # activate with Relu
                outs.append(acts[index] * (acts[index] > 0))
            prev_out = outs[index]
        # Compute the error signal
        predicted_value = np.argmax(outs[-1])
        # Count the total number of correct classifications
        confusion_matrix[int(correct_value), (predicted_value + 1)] += 1

    print(np.sum(np.diagonal(confusion_matrix)))
    print("Macro F1 " + str(macro_f1(confusion_matrix)))
    print("Micro F1 " + str(micro_f1(confusion_matrix)))
    print(confusion_matrix)
    return confusion_matrix, error


def micro_f1(confusion_matrix):
    """ Calculates the micro-F1 of the given confusion matrix.
            Input:
                confusion_matrix: The output of our classifier as a confusion matrix.
            Output:
                the micro_f1 score. Prints precision and recall if required.
    """

    # precision and recall is equal
    # implement per emotion precision and equal for results
    # for i in range(len(confusion_matrix[0])):
    #    print("Prec for " + LABEL2EMOTION[i] + " is")
    #    print(confusion_matrix[i][i] / np.sum(confusion_matrix[i, :]))
    #    print("Recall for " + LABEL2EMOTION[i] + " is")
    #    print(confusion_matrix[i][i] / np.sum(confusion_matrix[:, i]))
    precision = (np.sum(np.diagonal(confusion_matrix))) / (np.sum(confusion_matrix))
    recall = (np.sum(np.diagonal(confusion_matrix))) / (np.sum(confusion_matrix))
    return (2 * precision * recall) / (precision + recall)


def macro_f1(confusion_matrix):
    """ Calculates the macro-F1 of the given confusion matrix.
            Input:
                confusion_matrix: The output of our classifier as a confusion matrix.
            Output:
                the macro_f1 score. Prints precision and recall if required.
    """

    macro_prec = 0
    macro_rec = 0
    n_classes = len(confusion_matrix[0])
    for i in range(n_classes):
        macro_prec += (confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])) / n_classes
        macro_rec += (confusion_matrix[i, i] / np.sum(confusion_matrix[:, i])) / n_classes
    print("Macro_prec " + str(macro_prec))
    print("Macro_rec " + str(macro_rec))
    return (2 * macro_prec * macro_rec) / (macro_prec + macro_rec)


def save_ff_backprop_training_results_dynamic(weights, biases, extra):
    """ Saves the training results of the neural network into npy files """
    global run_counter
    for i in range(len(weights)):
        np.save((DIRECTORIES["weights"] + "w_" + str(i) + "_" + str(extra) + "_" + str(run_counter) + ".npy"), weights[i])
        np.save((DIRECTORIES["weights"] + "bias_" + str(i) + "_" + str(extra) + "_" + str(run_counter) + ".npy"), biases[i])
    run_counter += 1


def load_ff_backprop_training_results_dynamic(hidden_layers, extra):
    """ Loads the training results from npy files
            Input:
                hidden_layers: The Number of layers of our trained network
            Output:
                The weights and biases loaded from files.
    """

    weights = []
    biases = []
    for i in range(hidden_layers+1):
        weights.append(np.load(DIRECTORIES["weights"] + "w_" + str(i) + "_" + str(extra) + ".npy"))
        biases.append(np.load(DIRECTORIES["weights"] + "bias_" + str(i) + "_" + str(extra) + ".npy"))
    return weights, biases


def plot_errors(errors):
    """ Plots the validation errors using pyplot.
            Input:
                errors: the errors to plot.
    """

    plt.plot(errors)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Error')
    plt.title('Average Validation error per epoch')
    plt.show()


def generate_initial_test_set(feature_type, run_name):
    """ Generates the test set of the specific feature type based on the initial input. """

    test_feat, test_labels = input_data(feature_type, run_name, existing=False, sample=False, overwrite=False,
                                        two_class=False, hierarchical=False, initial=True)
    return test_feat, test_labels


def generate_initial_test_set_single_layer(feature_type, run_name, two_class):
    """ Generates the test set of the specific feature type based on the initial input For 1 layer of Hier. """

    test_feat, test_labels = input_data(feature_type, run_name, existing=False, sample=True, overwrite=False,
                                        two_class=two_class, hierarchical=True, initial=True)
    return test_feat, test_labels


def perform_flat_classification(feature_type, run_name, activation_function, eta, epoch, layers_list):
    """  Trains and performs the flat classification for our data set.
            Input:
                feature_type: The type of nn feature ("word2vec", "Bag_of_Words", "TF_IDF") .
                run_name: The name given to the particular run.
                activation_function_list : The activation function for our classifier.
                eta: The learning rate of our classifier.
                epoch: The epoch of our classifer
                layers_List: The list of number of neurons on our hidden layers
            Output:
                returns correct and cross entropy error. Print to console from other calls.
    """

    test_feat, test_labels = generate_initial_test_set(feature_type, run_name)
    print("Reading training features")
    train_feat, train_labels, val_feat, val_labels = input_data(feature_type, run_name, existing=False, sample=True,
        overwrite=False, two_class=False, hierarchical=False, initial=False)
    print("Training neural network...")
    weights, bias_weights = train_ff_backprop_nn_dynamic(eta, epoch, train_feat, train_labels, val_feat, val_labels,
                                                         TOTAL_CLASSES, activation_function, layers_list)

    save_ff_backprop_training_results_dynamic(weights, bias_weights, "flat")
    print("Classifying test data..")
    # weights, biases = load_ff_backprop_training_results_dynamic(1)
    conf_matrix, error = classify_with_ff_backprop_nn_dynamic(test_feat, test_labels, weights, bias_weights, TOTAL_CLASSES, activation_function)
    correct = np.sum(np.diagonal(conf_matrix))
    classification_results.append(correct)
    return correct, error


def perform_run_on_single_layer(feature_type, run_name, activation_function, eta, epoch, layers_list, top):
    """  Trains and tests a single layer of our hierarchical class.
            Input:
                feature_type: The type of nn feature ("word2vec", "Bag_of_Words", "TF_IDF") .
                run_name: The name given to the particular run.
                activation_function_list : The activation function for our classifier.
                eta: The learning rate of our classifier.
                epoch: The epoch of our classifer
                layers_List: The list of number of neurons on our hidden layers
            Output:
                 returns correct and cross entropy error. Print to console from other calls.
    """
    n_classes = 3
    test_feat, test_labels = generate_initial_test_set_single_layer(feature_type, run_name, top)
    print("Reading training features")
    if top:
        n_classes = 2
        train_feat, train_labels, val_feat, val_labels = input_data(feature_type, run_name, existing=False, sample=True,
            overwrite=False, two_class=True, hierarchical=True, initial=False)
    else:
        train_feat, train_labels, val_feat, val_labels = input_data(feature_type, run_name, existing=False, sample=True,
                                                                    overwrite=False, two_class=False, hierarchical=True,
                                                                    initial=False)

    print("Training neural network...")
    weights, bias_weights = train_ff_backprop_nn_dynamic(eta, epoch, train_feat, train_labels, val_feat, val_labels,
                                                         n_classes, activation_function, layers_list)

    # save_ff_backprop_training_results_dynamic(weights, bias_weights)
    print("Classifying test data..")
    # weights, biases = load_ff_backprop_training_results_dynamic(1)
    conf_matrix, error = classify_with_ff_backprop_nn_dynamic(test_feat, test_labels, weights, bias_weights, n_classes, activation_function)
    correct = np.diagonal(conf_matrix)
    classification_results.append(correct)
    return correct, error


def perform_hierarchical_classification(feature_type, run_name, activation_function_list, eta_list, epoch_list, layers_list):
    """  Trains and performs the hierarchical classification for our data set.
            Input:
                feature_type: A list of types of nn features ("word2vec", "Bag_of_Words", "TF_IDF") .
                run_name: The name given to the particular run.
                activation_function_list : The activation function for each classifier in a list.
                eta_list: The learning rate of each classifier in a list.
                epoch_list: The epoch of each classifier in a list
                layers_List: The neurons of Hidden L. of each classifier within a LIST of LISTS.
            Output:
                 returns correct and cross entropy error. Print to console from other calls.
    """

    print("Reading training features")
    test_feat_first_layer, test_labels = generate_initial_test_set(feature_type[0], run_name)
    if feature_type[0] == feature_type[1]:
        test_feat_second_layer = test_feat_first_layer
    else:
        test_feat_second_layer, test_labels = generate_initial_test_set(feature_type[1], run_name)
    binary_train_feat, binary_train_labels, binary_val_feat, binary_val_labels = input_data(feature_type[0], run_name,
        existing=False, sample=True, overwrite=False, two_class=True, hierarchical=True, initial=False)
    others_train_feat, others_train_labels, others_val_feat, others_val_labels = input_data(feature_type[1], run_name,
        existing=False, sample=True, overwrite=False, two_class=False, hierarchical=True, initial=False)

    print("Training both neural networks...")

    weights_list = []
    bias_weights_list = []

    x, y = train_ff_backprop_nn_dynamic(eta_list[0], epoch_list[0], binary_train_feat, binary_train_labels,
                                     binary_val_feat, binary_val_labels, 2, activation_function_list[0], layers_list[0])
    save_ff_backprop_training_results_dynamic(x, y, "HierFirst")

    weights_list.append(x)
    bias_weights_list.append(y)

    x, y = train_ff_backprop_nn_dynamic(eta_list[1], epoch_list[1], others_train_feat, others_train_labels,
        others_val_feat, others_val_labels, 3, activation_function_list[1], layers_list[1])
    save_ff_backprop_training_results_dynamic(x, y, "HierSecond")

    weights_list.append(x)
    bias_weights_list.append(y)

    print("Classifying test data through Hierch.")

    conf_matrix, error = classify_with_ff_backprop_nn_dynamic_hierarchical(test_feat_first_layer,
        test_feat_second_layer, test_labels, weights_list, bias_weights_list, activation_function_list)
    correct = np.sum(np.diagonal(conf_matrix))
    classification_results.append(correct)
    return correct, error


def classify_unseen_data_flat(weights, bias_weights, test_data_path, feature_type, activation_function, default=True):
    """  Classifies unseen data using a flat model with the weights and biases and misc provided."""

    test_conversations, test_labels = generate_initial_test_set(feature_type, "test")
    if not default:
        test_conversations = pre_process_data(test_data_path, "test")
        test_conversations = pre_process_unseen(feature_type, test_conversations)
        test_labels = np.zeros(np.shape(test_conversations)[1])
    print(test_conversations.shape)
    conf_matrix, error = classify_with_ff_backprop_nn_dynamic(test_conversations, test_labels, weights, bias_weights,
                                                              4, activation_function)


def classify_unseen_data_hier(weights_list, bias_weights_list, test_data_path, feature_type, activation_function_list, default=True):
    """  Classifies unseen data using a hier with the weights and biases provided and misc provided. """

    test_conversations, test_labels = generate_initial_test_set(feature_type, "test")
    if not default:
        test_conversations = pre_process_data(test_data_path, "test")
        test_conversations = pre_process_unseen(feature_type, test_conversations)
        test_labels = np.zeros(np.shape(test_conversations)[1])
    print(test_conversations.shape)
    conf_matrix, error = classify_with_ff_backprop_nn_dynamic_hierarchical(test_conversations,
                                                                           test_conversations, test_labels,
                                                                           weights_list, bias_weights_list,
                                                                           activation_function_list)


def main():
    """
        The main function, executes all the code from here. Alter this function according to the experiment
        to be conducted. Use the documentation of other functions to call correctly.
        List of functions to be called:
        1. perform_hierarchical_classification
        2. perform_flat_classification
        3. perform_run_on_single_layer
        4. classify_unseen_data_flat
        5. classify_unseen_data_hier
    """

    run_name = "TRIAL RUNS "

    # Uncomment, comment and edit accordingly to do desired runs. Use Documentation if necessary.

    # Loading Hier training results
    weights = []
    bias = []
    #  x, y = load_ff_backprop_training_results_dynamic(2, "tfidf_hier_first")
    x, y = load_ff_backprop_training_results_dynamic(2, "bow_hier_first")
    weights.append(x)
    bias.append(y)
    #  x, y = load_ff_backprop_training_results_dynamic(2, "tfidf_hier_second")
    x, y = load_ff_backprop_training_results_dynamic(2, "bow_hier_second")
    weights.append(x)
    bias.append(y)

    # Loading flat Training results
    # weights, bias = load_ff_backprop_training_results_dynamic(2, "tfidf_flat")
    # weights, bias = load_ff_backprop_training_results_dynamic(2, "bow_flat")

    # Classify data, flat or hier. If Dev, use True as last param.
    # classify_unseen_data_flat(weights, bias, DEV_DATA_PATH, "TF_IDF", "relu")
    # classify_unseen_data_flat(weights, bias, DEV_DATA_PATH, "Bag_of_Words", "relu")
    classify_unseen_data_hier(weights, bias, DEV_DATA_PATH, "Bag_of_Words", ["relu", "relu"])
    # classify_unseen_data_hier(weights, bias, DEV_DATA_PATH, "TF_IDF", ["relu", "relu"])
    # classify_unseen_data_hier(weights, bias, DEV_DATA_PATH, "TF_IDF", ["relu", "relu"], False)  # Dev set

    # comment, uncomment and add according to run desired. Use documentation if needed. CTRL + F Classifying for Res.
    # correct, error = perform_flat_classification(feature_type="word2vec", run_name=run_name,
    # activation_function="sigmoid", eta=0.04, epoch=50, layers_list=[])
    # correct, error = perform_flat_classification(feature_type="TF_IDF", run_name=run_name,
    #   activation_function="relu", eta=0.14, epoch=50, layers_list=[80, 80])
    # correct2, error2 = perform_hierarchical_classification(feature_type=["Bag_of_Words", "Bag_of_Words"],
    #   run_name=run_name, activation_function_list=["relu", "relu"], eta_list=[0.14, 0.14], epoch_list=[50, 50],
    #   layers_list=[[120, 60], [120, 60]])

    # for errors in validation_errors:
        # plot_errors(errors)


if __name__ == '__main__':
    main()
