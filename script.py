import os
import argparse
from PyPDF2 import PdfReader
import pickle
import shutil
from belt_nlp.bert_classifier_with_pooling import BertClassifierWithPooling
import config as ctg
import csv

def extract_text_from_pdfs(pdf_paths):
    #Extracxt texts(str) from a list of PDF
    x_test = []
    for pdf_path in pdf_paths:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            x_test.append(text)
    return x_test

def get_pdf_files_from_directory(directory_path):
    #A fuction that finds the path of every PDF file in a directory
    pdf_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from PDF files in a directory.")
    parser.add_argument('directory', default='dataset\data\data\ACCOUNTANT_2', type=str, help="Path to the directory containing PDF files.")
    
    args = parser.parse_args()
    directory_path = args.directory #Get Directory Path from command line
    # directory_path = 'dataset\data\data\ACCOUNTANT_2'

    with open(ctg.lable_mapping_dir, 'rb') as fp:
        labels_mapping = pickle.load(fp)

    # Get all PDF files from the specified directory
    pdf_files = get_pdf_files_from_directory(directory_path)
    file_name = [os.path.basename(path) for path in pdf_files]

    # Extract text from each PDF
    x_test = extract_text_from_pdfs(pdf_files)

    MODEL_PARAMS = ctg.MODEL_PARAMS

    model = BertClassifierWithPooling(**MODEL_PARAMS)
    model.load(ctg.model_dir)

    y_pred = model.predict(x_test).tolist()
    predictions = [labels_mapping[x] for x in y_pred]


    for source, destination, file_name in zip(pdf_files, predictions, file_name):
        destination_dir = f'{ctg.output_dir}{destination}/{file_name}'
        # Check if the destination directory exists, if not create it
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        # Move the file from source to destination
        try:
            shutil.move(source, destination_dir)
            print(f"Moved file from {source} to {destination_dir}")
        except Exception as e:
            print(f"Failed to move {source} to {destination}: {e}")

    
    with open(ctg.output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "category"])  # Writing header
        for value1, value2 in zip(file_name, predictions):
            writer.writerow([value1, value2])