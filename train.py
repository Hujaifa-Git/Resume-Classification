from datasets import load_from_disk
from belt_nlp.bert_classifier_with_pooling import BertClassifierWithPooling
import pickle
import torch
import numpy as np
import config as ctg


if __name__ == "__main__":
    #Load Datasets and Lable Mappings
    train_ds = load_from_disk(ctg.train_data_dir)
    test_ds = load_from_disk(ctg.test_data_dir)
    with open(ctg.lable_mapping_dir, 'rb') as fp:
        labels_mapping = pickle.load(fp)

    x_train, y_train = train_ds['Resume_str'], train_ds['Category']
    x_test, y_test = test_ds['Resume_str'], test_ds['Category']

    MODEL_PARAMS = ctg.MODEL_PARAMS

    #Load BERT and start training
    model = BertClassifierWithPooling(**MODEL_PARAMS)
    model.fit(x_train, y_train)
    classes = model.predict(x_test)
    accurate = sum(classes == np.array(y_test))
    accuracy = accurate / len(y_test)
    print(f"Test accuracy: {accuracy}")
    model.save(ctg.model_dir)

