import os
import numpy as np
from collections import Counter
from string import punctuation


def review_parser(folder):
    feature_vector = [1, 1, 1, 1, 1, 1, 1, 1]
    num_files = 0
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            with open(os.path.join(folder, filename)) as f:
                word_freq = Counter([word.strip(punctuation) for line in f for word in line.split()])
                feature_vector[0] += 1 if word_freq['awful'] > 0 else 0
                feature_vector[1] += 1 if word_freq['bad'] > 0 else 0
                feature_vector[2] += 1 if word_freq['boring'] > 0 else 0
                feature_vector[3] += 1 if word_freq['dull'] > 0 else 0
                feature_vector[4] += 1 if word_freq['effective'] > 0 else 0
                feature_vector[5] += 1 if word_freq['enjoyable'] > 0 else 0
                feature_vector[6] += 1 if word_freq['great'] > 0 else 0
                feature_vector[7] += 1 if word_freq['hilarious'] > 0 else 0
        
            num_files += 1

    #normalize the vector
    for i in range(len(feature_vector)):
        feature_vector[i] = feature_vector[i] / (num_files + 1)

    return feature_vector


neg_vector = review_parser('./txt_sentoken/neg/')
pos_vector = review_parser('./txt_sentoken/pos/')

print("\nQ6:\n")

print('Positive Vector:', pos_vector)
print('Negative Vector:', neg_vector)
print("\nQ7:\n")



def classifier(filepath, pos_vector, neg_vector):
    test_instance = [0, 0, 0, 0, 0, 0, 0, 0]
    if filepath.endswith('.txt'):
        with open(filepath) as f:
            word_freq = Counter([word.strip(punctuation) for line in f for word in line.split()])
            test_instance[0] += 1 if word_freq['awful'] > 0 else 0
            test_instance[1] += 1 if word_freq['bad'] > 0 else 0
            test_instance[2] += 1 if word_freq['boring'] > 0 else 0
            test_instance[3] += 1 if word_freq['dull'] > 0 else 0
            test_instance[4] += 1 if word_freq['effective'] > 0 else 0
            test_instance[5] += 1 if word_freq['enjoyable'] > 0 else 0
            test_instance[6] += 1 if word_freq['great'] > 0 else 0
            test_instance[7] += 1 if word_freq['hilarious'] > 0 else 0
    
    return np.dot(test_instance, pos_vector), np.dot(test_instance, neg_vector)

pos_prob, neg_prob = classifier('./txt_sentoken/pos/cv003_11664.txt', pos_vector=pos_vector, neg_vector=neg_vector)

print("probability ./txt_sentoken/pos/cv003_11664.txt is a positive review:", pos_prob)
print("probability ./txt_sentoken/pos/cv003_11664.txt is a negative review:", neg_prob)

print('\n')

pos_prob, neg_prob = classifier('./txt_sentoken/neg/cv008_29326.txt', pos_vector=pos_vector, neg_vector=neg_vector)

print("probability ./txt_sentoken/neg/cv008_29326.txt is a positive review:", pos_prob)
print("probability ./txt_sentoken/neg/cv008_29326.txt is a negative review:", neg_prob)

print('\n')

print('this one is armageddon, so the review has words like \'hilarious\' written in a negative way, and the only mention of \'great\' is in reference to the movie jaws, so the classifier gets this one wrong')
pos_prob, neg_prob = classifier('./txt_sentoken/neg/cv070_13249.txt', pos_vector=pos_vector, neg_vector=neg_vector)

print("probability ./txt_sentoken/neg/cv070_13249.txt is a positive review:", pos_prob)
print("probability ./txt_sentoken/neg/cv070_13249.txt is a negative review:", neg_prob)

print("\nQ8:\n")

def tester(folder_of_databases, pos_vector, neg_vector):
    confusion_matrix = [0, 0, 0, 0]
    pos_directory = os.path.join(folder_of_databases, 'pos')
    neg_directory = os.path.join(folder_of_databases, 'neg')

    #test both the positive and negative reviews
    for folder in (pos_directory, neg_directory):
        for filename in os.listdir(folder):
            if filename.endswith('.txt'):
                with open(os.path.join(folder, filename)) as f:
                    word_freq = Counter([word.strip(punctuation) for line in f for word in line.split()])

                    test_instance = [0, 0, 0, 0, 0, 0, 0, 0]

                    test_instance[0] += 1 if word_freq['awful'] > 0 else 0
                    test_instance[1] += 1 if word_freq['bad'] > 0 else 0
                    test_instance[2] += 1 if word_freq['boring'] > 0 else 0
                    test_instance[3] += 1 if word_freq['dull'] > 0 else 0
                    test_instance[4] += 1 if word_freq['effective'] > 0 else 0
                    test_instance[5] += 1 if word_freq['enjoyable'] > 0 else 0
                    test_instance[6] += 1 if word_freq['great'] > 0 else 0
                    test_instance[7] += 1 if word_freq['hilarious'] > 0 else 0

                    pos_prob, neg_prob = np.dot(test_instance, pos_vector), np.dot(test_instance, neg_vector)

                    if pos_prob > neg_prob:
                        if folder == pos_directory:
                            confusion_matrix[0] += 1 #TP
                        else:
                            confusion_matrix[1] += 1 #FP
                    if neg_prob > pos_prob:
                        if folder != neg_directory:
                            confusion_matrix[2] += 1 #FN
                        else:
                            confusion_matrix[3] += 1 #TN
    
    total = sum(confusion_matrix)
    
    confusion_matrix[0] = confusion_matrix[0]/total
    confusion_matrix[1] = confusion_matrix[1]/total
    confusion_matrix[2] = confusion_matrix[2]/total
    confusion_matrix[3] = confusion_matrix[3]/total

    return confusion_matrix, confusion_matrix[0] + confusion_matrix[3]

conf_matrix, accuracy = tester('./txt_sentoken', pos_vector, neg_vector)
print('Confusion Matrix:', conf_matrix, '\nClassification Accuracy:', accuracy)