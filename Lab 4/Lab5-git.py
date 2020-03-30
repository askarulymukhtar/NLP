import torch
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel, BertConfig, BertPreTrainedModel, BertForPreTraining, BertForMaskedLM
import torch.nn as nn
from tqdm import tqdm, tqdm_notebook
import os
import random

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

RUBERT_PATH = '../../ml_models/ru_conversational_cased_L-12_H-768_A-12_pt'
modelpath = os.path.join(RUBERT_PATH,'pytorch_model.bin')



#!pip install transformers #huggingface



os.path.isfile(os.path.join(RUBERT_PATH,'pytorch_model.bin'))


# tokenizer = BertTokenizer.from_pretrained(os.path.join(RUBERT_PATH,'vocab.txt'))
tokenizer = BertTokenizer.from_pretrained(RUBERT_PATH, do_lower_case=False)
config = BertConfig.from_json_file(os.path.join(RUBERT_PATH,'bert_config.json'))
bert = BertForPreTraining.from_pretrained(modelpath, config=config)
# model.eval()

len(tokenizer.vocab)


max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

def get_means(sentence):
    tokenized_text = tokenizer.tokenize(sentence)
    tokenized_text = tokenized_text[:max_input_length-2]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    
    segments_ids = [1] * len(tokenized_text)    
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    predictions = bert(tokens_tensor, token_type_ids=segments_tensors)
    _, secondDims, thirdDims = predictions[0].shape
    
    finalVector = []
    
    for i in range(secondDims):
        currentArr = predictions[0][0][i].detach().numpy()
        if len(finalVector) == 0:
            finalVector = currentArr
        else:
            finalVector = np.add(finalVector, currentArr)
    return np.mean(finalVector)



#texts data
texts_df = pd.read_csv('texts_train.txt', sep="\t", header=None)
texts_df.columns = ["text"]
texts_df.head()



get_means(texts_df['text'][0])



#scores data
if (os.path.isfile('collected_data.csv')):
    scores_df = pd.read_csv('collected_data.csv', dtype='float64')
else:
    scores_df = pd.read_csv('scores_train.txt', sep="\t", header=None, dtype='float64')
    scores_df.columns = ["tonality"]
    vector_means = [get_means(sentence) for sentence in texts_df["text"].tolist()]
    scores_df['vector_means'] = vector_means

scores_df.head()



scores_df.to_csv('collected_data.csv', index = False, header=True)



tone_levels = np.array(scores_df['tonality'])
features = np.array(scores_df['vector_means']).reshape(-1, 1)



# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test  = train_test_split(features, tone_levels, test_size = 0.2, random_state = random.seed(SEED))

print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test .shape)



from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(n_estimators=100,
                                  learning_rate=0.1,
                                  random_state=random.seed(SEED),
                                  loss='ls',
                                  max_depth=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)



def get_mae(pred, test):
    # Calculate the absolute errors
    errors = abs(pred - test)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    return errors



errors = get_mae(y_pred, y_test)



def get_accuracy(errs, test_data):
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errs / test_data)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    return accuracy



get_accuracy(errors, y_test)



from sklearn import metrics
y_pred = [int(item) for item in y_pred]
print(metrics.classification_report(y_test, y_pred))




from sklearn.svm import SVC

clf = SVC(gamma='auto')
clf.fit(X_train, y_train)
clf_y_pred = clf.predict(X_test)



clf_errors = get_mae(clf_y_pred, y_test)



get_accuracy(clf_errors, y_test)



clf_y_pred = [int(item) for item in clf_y_pred]
print(metrics.classification_report(y_test, clf_y_pred))




from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
knn_y_pred = knn.predict(X_test)



knn_errors = get_mae(knn_y_pred, y_test)




get_accuracy(knn_errors, y_test)



knn_y_pred = [int(item) for item in knn_y_pred]
print(metrics.classification_report(y_test, clf_y_pred))