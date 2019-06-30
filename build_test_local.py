#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import numpy
import pickle
import random

# test_file = '/home/cl67/workspace/se/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/query_asin_test.data'
#
# train_file = '/home/cl67/workspace/se/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/query_asin_train.data'
#
# vocab_file = '/home/cl67/workspace/se/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/query_asin.vocab'
#
# test_knn = '/home/cl67/workspace/se/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/test_kdd.pickle'
#
# test_qlist = '/home/cl67/workspace/se/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/test_qlist.pickle'
#
# test_alist = '/home/cl67/workspace/se/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/test_alist.pickle'


# # path original test data of query-title pair
# # test_file = 'https://yiyangxubucket.s3.amazonaws.com/summer2019/pythonfiles/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/query_asin_test.data'
# test_file = "s3://yiyangxubucket/summer2019/pythonfiles/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/query_asin_test.data"
#
# # path original train data of query-title pair
# train_file = 'https://yiyangxubucket.s3.amazonaws.com/summer2019/pythonfiles/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/query_asin_train.data'
#
# # path of vocabulary built
# vocab_file = 'https://yiyangxubucket.s3.amazonaws.com/summer2019/pythonfiles/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/query_asin.vocab'
#
# # path of all query-title pairs
# # for easy access when generating ground-false (no purchase), or easy access for checking existing pairs
# test_knn = 'https://yiyangxubucket.s3.amazonaws.com/summer2019/pythonfiles/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/test_kdd.pickle'
#
# # path of query-word-vector-representation (dictionary) built
# test_qlist = 'https://yiyangxubucket.s3.amazonaws.com/summer2019/pythonfiles/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/test_qlist.pickle'
#
# # path of title-word-vector-representation (dictionary) built
# test_alist = 'https://yiyangxubucket.s3.amazonaws.com/summer2019/pythonfiles/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/test_alist.pickle'
#
# train_qlist = 'https://yiyangxubucket.s3.amazonaws.com/summer2019/pythonfiles/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/train_qlist.pickle'
#
# train_alist = 'https://yiyangxubucket.s3.amazonaws.com/summer2019/pythonfiles/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/train_alist.pickle'
#
# # the path to complete sparse testing data that matches torch.Dataset features
# test_file_sparse = 'https://yiyangxubucket.s3.amazonaws.com/summer2019/pythonfiles/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/test_sparse_data.txt'
#
# # the path to complete sparse training data that matches torch.Dataset features
# train_file_sparse = 'https://yiyangxubucket.s3.amazonaws.com/summer2019/pythonfiles/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/train_sparse_data.txt'
#


# path original test data of query-title pair
# test_file = 'https://yiyangxubucket.s3.amazonaws.com/summer2019/pythonfiles/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/query_asin_test.data'
test_file = '/Users/yiyangxu/Library/Mobile Documents/com~apple~CloudDocs/Research/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/query_asin_test_data.txt'

# path original train data of query-title pair
train_file = '/Users/yiyangxu/Library/Mobile Documents/com~apple~CloudDocs/Research/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/query_asin_train_data.txt'

# path of vocabulary built
vocab_file = '/Users/yiyangxu/Library/Mobile Documents/com~apple~CloudDocs/Research/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/query_asin_vocab.txt'

# path of all query-title pairs
# for easy access when generating ground-false (no purchase), or easy access for checking existing pairs
test_knn = '/Users/yiyangxu/Library/Mobile Documents/com~apple~CloudDocs/Research/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/test_kdd_pickle.txt'

# path of query-word-vector-representation (dictionary) built
test_qlist = '/Users/yiyangxu/Library/Mobile Documents/com~apple~CloudDocs/Research/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/test_qlist_pickle.txt'

# path of title-word-vector-representation (dictionary) built
test_alist = '/Users/yiyangxu/Library/Mobile Documents/com~apple~CloudDocs/Research/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/test_alist_pickle.txt'

train_qlist = '/Users/yiyangxu/Library/Mobile Documents/com~apple~CloudDocs/Research/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/train_qlist_pickle.txt'

train_alist = '/Users/yiyangxu/Library/Mobile Documents/com~apple~CloudDocs/Research/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz//train_alist_pickle.txt'

# the path to complete sparse testing data that matches torch.Dataset features
test_file_sparse = '/Users/yiyangxu/Library/Mobile Documents/com~apple~CloudDocs/Research/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/test_sparse_data.txt'

# the path to complete sparse training data that matches torch.Dataset features
train_file_sparse = '/Users/yiyangxu/Library/Mobile Documents/com~apple~CloudDocs/Research/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/train_sparse_data.txt'

# randomly generated non-query-title pairs
test_random_non_query_title_pair = '/Users/yiyangxu/Library/Mobile Documents/com~apple~CloudDocs/Research/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/test_random_non_query_title_pairs.txt'
train_random_non_query_title_pair = '/Users/yiyangxu/Library/Mobile Documents/com~apple~CloudDocs/Research/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/train_random_non_query_title_pairs.txt'

# complete 1 and 0 labelled data
test_file_sparse_whole = '/Users/yiyangxu/Library/Mobile Documents/com~apple~CloudDocs/Research/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/test_sparse_whole_data.txt'

# complete 1 and 0 labelled data
train_file_sparse_whole = '/Users/yiyangxu/Library/Mobile Documents/com~apple~CloudDocs/Research/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/train_sparse_whole_data.txt'


# Extract word in a sentence to form a list without stop words
def word_extraction(sentence):
    """
    This extrace words from a single sentence and return a list.
    :param sentence: String
    :return: list
    """
    # set(stopwords.words('english'))
    ignore = ['a', "the", "is"]
    words = re.sub("[^\w]", " ", sentence).split()
    cleaned_text = [w.lower() for w in words if w not in ignore]
    return cleaned_text


def tokenize(sentences):
    """
    This function takes in sentences and extract all the word tokens calling word_extraction().
    :param sentences: String
    :return: list of words
    """
    words = []
    for sentence in sentences:
        w = word_extraction(sentence)
        words.extend(w)
    words = sorted(list(set(words)))
    return words


def generate_bow(allsentences):
    """
    This function generates and displays the vocabulary(words) of all query/produce Strings
    :param allsentences:
    :return: the whole vocabulary
    """
    vocab = tokenize(allsentences)
    return vocab
    print("Word List for Document \n{0} \n".format(vocab));


# transfer the words list to sparse format
def words2sparse(vocab, words):
    """
    Given a String of words converts the String to continuous vector representation according to its word frequencies.
    :param vocab:
    :param words:
    :return:
    """
    word_sparse = {}
    for w in words:
        for i, word in enumerate(vocab):
            if word == w:
                if i in word_sparse:
                    word_sparse[i] += 1
                else:
                    word_sparse[i] = 1
    return word_sparse


# The experiment for having get all the data
# for sentence in sentences:
def write_to_sparse(test_query_sentences, test_title_sentences, file_path, test_query_label, vocab):
    """
    This is for generating the file that convert the query product pairs to feature(vocabulary) and label(0/1) data,
    which is taken as input by TORCH.UTIL.DATA.DATASET (class AmazonDataset()) before loaded into dataloader
    :param test_query_sentences:
    :param test_title_sentences:
    :param file_path:
    :param test_query_label:
    :param vocab:
    :return:
    """
    # generating the sparse format data for the training and testing
    f = open(file_path, 'w')
    data_size = len(test_query_sentences)
    feature_dim = len(vocab)

    f.write("%d,%d\n" % (data_size, feature_dim))

    for j in range(data_size):
        query_sentence = test_query_sentences[j]
        title_sentence = test_title_sentences[j]

        query_words = word_extraction(query_sentence)
        title_words = word_extraction(title_sentence)

        # print("{0},".format(test_query_label[j].replace('\n',''))),
        f.write("%s, " % test_query_label[j].replace('\n', ''))
        # print("{0},".format(test_query_label[j].replace('\n','')), f),

        query_words_sparse = words2sparse(vocab, query_words)
        title_words_sparse = words2sparse(vocab, title_words)

        for w in query_words_sparse:
            # print("{0}:{1}".format(w, query_words_sparse[w])),
            f.write("%s:%s " % (w, query_words_sparse[w]))
            # print("{0}:{1}".format(w, query_words_sparse[w]), f),

        # print(","),
        # print(",", f)
        f.write(",")
        for w in title_words_sparse:
            # print("{0}:{1}".format(w, title_words_sparse[w])),
            f.write("%s:%s " % (w, title_words_sparse[w]))
            # print("{0}:{1}".format(w, title_words_sparse[w]), f),
        # print("\n")
        f.write("\n")


def add_sparse(sentences, vocab):
    """
    Creates a dictionary that stores the word-embedding of each sentence.
    :param sentences:
    :param vocab:
    :return:
    """
    words_dict = {}

    for sentence in sentences:

        if sentence not in words_dict:
            words = word_extraction(sentence)  # extract words

            words_sparse = words2sparse(vocab, words)  # compare to vocabulary

            words_dict[sentence] = words_sparse

    return words_dict


allsentences = []
test_query_sentences = []
test_title_sentences = []
test_query_label = []

train_title_sentences = []
train_query_sentences = []
train_query_label = []

test_query_asin_list = {}
train_query_asin_list = {}
# the query-product pairs from both testing file and training file
test_train_query_asin_list = {}

print("Start reading test words")

with open(test_file) as tf:
    for line in tf:
        # print(line)
        items = line.split(',')

        # print(items)
        test_query_sentences.append(items[0])
        test_title_sentences.append(items[1])
        test_query_label.append(items[2])

        # Add out logic of generating dictionary
        # Record the existing matches of each pair of query to product title
        if items[0] in test_query_asin_list.keys():
            if items[1] not in test_query_asin_list[items[0]]:
                test_query_asin_list[items[0]].append(items[1])
        else:
            test_query_asin_list[items[0]] = []
            test_query_asin_list[items[0]].append(items[1])

        if items[0] in test_train_query_asin_list.keys():
            if items[1] not in test_train_query_asin_list[items[0]]:
                test_train_query_asin_list[items[0]].append(items[1])
        else:
            test_train_query_asin_list[items[0]] = []
            test_train_query_asin_list[items[0]].append(items[1])

# knn_file = open(test_knn,'wb')
# pickle.dump(test_query_asin_list,knn_file)  # save the query-product file on disk

print("Start reading training words")

with open(train_file) as tf:
    for line in tf:
        items = line.split(',')
        # Data cleaning for some wired data points
        if (len(items) > 2):
            train_query_sentences.append(items[0])
            train_title_sentences.append(items[1])
            train_query_label.append(items[2])
        # Add out logic of generating dictionary
        # Record the existing matches of each pair of query to product title
        if items[0] in train_query_asin_list.keys():
            if items[1] not in train_query_asin_list[items[0]]:
                train_query_asin_list[items[0]].append(items[1])
        else:
            train_query_asin_list[items[0]] = []
            train_query_asin_list[items[0]].append(items[1])

        if items[0] in test_train_query_asin_list.keys():
            if items[1] not in test_train_query_asin_list[items[0]]:
                test_train_query_asin_list[items[0]].append(items[1])
        else:
            test_train_query_asin_list[items[0]] = []
            test_train_query_asin_list[items[0]].append(items[1])

# all sentences i queries and product titles
allsentences.extend(test_query_sentences)
allsentences.extend(test_title_sentences)
allsentences.extend(train_query_sentences)
allsentences.extend(train_title_sentences)

sentences = ["Joe waited for the train train", "The train was late", "Mary and Samantha took the bus",
             "I looked for Mary and Samantha at the bus station",
             "Mary and Samantha arrived at the bus station early but waited until noon for the bus"]

print("Start buiding vocab ")

vocab = generate_bow(allsentences)

print("Start generating testing list")
# test_query_list = add_sparse(test_query_sentences,vocab)  # each query to its vector representation (dictionary)
# test_asin_list = add_sparse(test_title_sentences,vocab)  # each title to its vector representation (dictionary)

# qlist_file = open(test_qlist,'wb')
# pickle.dump(test_query_list,qlist_file)  # store query-vectors in the file on disk
# alist_file = open(test_alist,'wb')
# pickle.dump(test_asin_list,alist_file)  # store title-vectors in the file on disk


# # '''
# with open (vocab_file, 'w') as vf:
#     for w in vocab:
#         vf.write("%s\n" % (w))
#
# write_to_sparse(test_query_sentences,test_title_sentences, test_file_sparse, test_query_label, vocab)
# write_to_sparse(train_query_sentences,train_title_sentences, train_file_sparse, train_query_label, vocab)


# print(test_query_asin_list)


# # set of all queries
# test_train_query_set = set(test_query_sentences + train_query_sentences)
# # set of all titles
# test_train_title_set = set(test_title_sentences + train_title_sentences)
# set of all queries
test_train_query_set = list(set(test_query_sentences + train_query_sentences))
# set of all titles
test_train_title_set = list(set(test_title_sentences + train_title_sentences))


# generate non-purchase data, append it to the existing file
# randomly generate non query-product pair from "test_query_asin_list"
def random_non_query_title(file_path, sparse_file_path, new_sparse_file_path, test_train_query_asin_dic,
                           test_train_query, test_train_title, vocab):
    f_new = open(new_sparse_file_path, 'w')
    f_sparse = open(sparse_file_path, 'r')
    f = open(file_path, 'w')

    sizes = f_sparse.readline().split(',')

    size = int(sizes[0])
    new_data_size = size + int(sizes[0])
    feature_size = int(sizes[1])

    f_new.write("%s,%s" % (new_data_size, feature_size))
    f_new.write("\n")

    i = 0
    while i < 2 * size:
        if i % 2 == 0:
            oneline = []  # because writelines() takes a list
            oneline.append(f_sparse.readline())
            f_new.writelines(oneline)
            i = i + 1
        else:
            random_query = random.choice(test_train_query)
            random_title = random.choice(test_train_title)
            if random_title not in test_train_query_asin_dic[random_query]:
                query_words = word_extraction(random_query)
                title_words = word_extraction(random_title)

                query_words_sparse = words2sparse(vocab, query_words)
                title_words_sparse = words2sparse(vocab, title_words)

                f_new.write("%s, " % 0)
                for w in query_words_sparse:
                    # print("{0}:{1}".format(w, query_words_sparse[w])),
                    f_new.write("%s:%s " % (w, query_words_sparse[w]))
                    # print("{0}:{1}".format(w, query_words_sparse[w]), f),

                # print(","),
                # print(",", f)
                f_new.write(",")
                for w in title_words_sparse:
                    # print("{0}:{1}".format(w, title_words_sparse[w])),
                    f_new.write("%s:%s " % (w, title_words_sparse[w]))
                    # print("{0}:{1}".format(w, title_words_sparse[w]), f),
                # print("\n")
                f_new.write("\n")

                f.write("%s,%s,%s" % (random_query, random_title, 0))
                f.write("\n")
                i = i + 1


random_non_query_title(test_random_non_query_title_pair, test_file_sparse, test_file_sparse_whole,
                       test_train_query_asin_list, test_train_query_set, test_train_title_set, vocab)
random_non_query_title(train_random_non_query_title_pair, train_file_sparse, train_file_sparse_whole,
                       test_train_query_asin_list, test_train_query_set, test_train_title_set, vocab)
