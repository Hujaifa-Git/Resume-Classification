import torch

resume_dir = 'G:\Backup\dataset\Resume\Resume.csv'
figure_category_dir = 'figures/lables_distribution.jpg'
figure_length_dir = 'figures/token_length_distribution.jpg'
train_data_dir = 'train_data'
test_data_dir = 'test_data'
lable_mapping_dir = 'labels_mapping.pkl'

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PARAMS = {
        "num_labels": 24,
        "batch_size": 16,
        "learning_rate": 5e-5,
        "epochs": 1,
        "chunk_size": 510, #Each Chunk Size
        "stride": 480, #Offset after each chunk
        "minimal_chunk_length": 128,
        "maximal_text_length": 510 * 4,
        "pooling_strategy": "mean",
        "device": device,
        "many_gpus": True,
    }

model_dir = 'model_final_csv_classifier'
output_csv ='categorized_csv.csv'
output_dir = 'Output/'