# Post2Vec

## Data Download
- Data download [here [12.8GB]](https://drive.google.com/file/d/1g1tAebVnT76pYcY74IxyqoMU7KBEXPmb/view?usp=sharing).
- Or you can download raw data from [Stack Overflow data dump](https://archive.org/download/stackexchange), the data preprocessing scripts are provided [here](https://github.com/post2vec/post2vec/tree/master/src/data_preparation).

## Experiment Instructions

|       |           Command                                                                                                             |
|-------|-----------------------------------------------------------------------------------------------------------------------------------|
|       | Post2Vec                                                                                                                          |
| Train | ```time CUDA_VISIBLE_DEVICES=0 .env/bin/python3 src/approaches/post2vec/post2vec_train.py```            |
| Test  | ```time CUDA_VISIBLE_DEVICES=0 .env/bin/python3 src/approaches/post2vec/post2vec_test.py```     |
|       | TagCNN                                                                                                                            |
| Train | ```time CUDA_VISIBLE_DEVICES=0 .env/bin/python3 src/approaches/stoa/tagcnn/tagcnn_train.py```    |
| Test  | ```time CUDA_VISIBLE_DEVICES=0 .env/bin/python3 src/approaches/stoa/tagcnn/tagcnn_test.py```     |
|       | TagRCNN                                                                                                                           |
| Train | ```time CUDA_VISIBLE_DEVICES=5 .env/bin/python3 src/approaches/stoa/tagrcnn/tagrcnn_train.py``` |
| Test  | ```time CUDA_VISIBLE_DEVICES=0 .env/bin/python3 src/approaches/stoa/tagrcnn/tagrcnn_test.py```   |
