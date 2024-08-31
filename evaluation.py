from datasets import load_from_disk
from belt_nlp.bert_classifier_with_pooling import BertClassifierWithPooling
import pickle
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import config as ctg

def calculcate_performance(y_test, y_pred):
    # Accuracy Precision, Recall, F1-score (for multi-class)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')  # 'macro' calculates the precision per class and then averages them
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f'y true {y_test}')
    print(f'y pred {y_pred}')
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

if __name__ == "__main__":
    test_ds = load_from_disk(ctg.test_data_dir)
    with open(ctg.lable_mapping_dir, 'rb') as fp:
        labels_mapping = pickle.load(fp)

    x_test, y_test = test_ds['Resume_str'], test_ds['Category']

    MODEL_PARAMS = ctg.MODEL_PARAMS

    model = BertClassifierWithPooling(**MODEL_PARAMS)
    model.load(ctg.model_dir)

    y_pred = model.predict(x_test).tolist()

    calculcate_performance(y_test, y_pred)





