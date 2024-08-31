import pandas as pd
from datasets import Dataset, load_from_disk
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import config as ctg
from transformers import BertTokenizer

def create_category_graphs(label_df):
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    # Bar chart to visualize the frequency of category
    ax[0].bar(label_df['Category'], label_df['count'], color='blue', edgecolor='black')
    ax[0].set_title('Histogram of Counts')
    ax[0].set_xlabel('Category')
    ax[0].set_ylabel('Count')
    ax[0].tick_params(axis='x', rotation=90)  # Rotate x labels for better readability
    np.random.seed(0) 
    colors = np.random.rand(len(df), 3)  # Generate random colors for the Pie chart
    # Pie chart to visualize the frequency of category
    ax[1].pie(label_df['count'], labels=label_df['Category'], autopct='%1.1f%%', startangle=140, colors=colors)

    plt.tight_layout()
    plt.show()
    fig.savefig(ctg.figure_category_dir, format='jpg', dpi=300)


def token_length_graph(data):
    token_length_bins = ['512-', '512-1024', '1024-2048', '2048-4096', '4096+']
    token_length_count = [0,0,0,0,0]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for i in data:
        lenght = len(tokenizer(i['Resume_str'], truncation=False)['input_ids'])
        if lenght<=512:token_length_count[0]+=1
        elif lenght<=1024:token_length_count[1]+=1
        elif lenght<=2048:token_length_count[2]+=1
        elif lenght<=4096:token_length_count[3]+=1
        else:token_length_count[4]+=1
    print(token_length_count)

    fig, ax = plt.subplots()
    ax.bar(token_length_bins, token_length_count, edgecolor='black')
    ax.set_title('Histogram of Token Lengths')
    ax.set_xlabel('Range')
    ax.set_ylabel('Length')
    ax.tick_params(axis='x', rotation=90) 

    plt.tight_layout()
    plt.show()
    fig.savefig(ctg.figure_length_dir, format='jpg', dpi=300)
    exit()




if __name__ == "__main__":
    df = pd.read_csv(ctg.resume_dir) #Read Given CSV File

    label_df = df['Category'].value_counts().reset_index() #Number of Categories and their frequency

    print(label_df)

    create_category_graphs(label_df)

    ds = Dataset.from_pandas(df) #Convert DataFrame into Transformers Dataset Object
    token_length_graph(ds)

    #LableEncoder to convert strings of category into numerical value [0....n]
    label_encoder = LabelEncoder()
    label_encoder.fit(ds['Category'])
    labels = label_encoder.transform(ds['Category'])
    #Create a dictionary of category encodings and category names
    lables_mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
    ds = ds.map(lambda x, i: {'Category': labels[i]}, with_indices=True)
    ds = ds.map(lambda x: {'Resume_str': x['Resume_str'].lower()}) #Convert resumes into lower for better tokenization
    ds = ds.shuffle().train_test_split(test_size=0.1)

    train_ds = ds['train']
    test_ds = ds['test']

    #Save Datasets and lable mappings
    train_ds.save_to_disk(ctg.train_data_dir)
    train_ds.save_to_disk(ctg.test_data_dir)
    with open(ctg.lable_mapping_dir, 'wb') as fp:
        pickle.dump(lables_mapping, fp)

    print(ds)
