# ICD-codes Multi-Label classification with BERT
A script built with the [transformers](https://huggingface.co/docs/transformers/index) and [datasets](https://huggingface.co/docs/datasets/index) packages.
## Requirements
To use prepare the use of the script use the following
```bash
pip install -r requirements.txt
```
## Data format
The dataset must be in a ```.csv``` format with the <ins>text</ins> column named as ```full_note``` and the <ins>label</ins> column named as ```codes```. The text is expected in a ```string``` format and the labels are expected in ```string,string``` format.
## Configuration file
The configuration file is the place where the parameters of the training session can be defined. The total options are
```json
{
  "csv_data_path": "/path/to/the/csv/training/file.csv",
  "save_name": "/name_of_the_output_folder_within_the_model_path/",
  "model_path": "/path/to/the/BERT/model/and/tokenizer/",
  "parameters": {
    "epochs": 10,
    "learning_rate": 0.00002,
    "patience": 1,
    "batch": {
      "size": 4,
      "accumulation": 16
    }
  },
  "random_state": 123,
  "checkpoints": {
    "path": "/path/to/a/folder/containing/a/number/of/bert/checkpoints/",
    "flag": "checkpoint-in-name-flag-*"
  }
}
```
##Execution
The file accepts as an argument the path to the ```config.json``` file which by default will first look for a <ins>config</ins> file with that name within the same folder as the <ins>python</ins> file.

To execute the file with a different configuration file name or path use
```bash
python MultiLabelClassification.py -config_dir /path/to/config/file.json
```
##Output
The output will be saved in the specified folder and will report **Accuracy**, **Precision**, **Recall**, and **F1-score** for a separate test set, using 10% of the original dataset.
