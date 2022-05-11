# Cryptocurrency Bubble Detection: A New Stock Market Dataset, Financial Task & Hyperbolic Models

This is an official pytorch implementation of our NAACL 2022 paper Cryptocurrency Bubble Detection: A New Stock Market Dataset, Financial Task & Hyperbolic Models. In this repository, we provide PyTorch code for training our proposed MBHN model. We also provide scripts to develop data from the provided raw data. 

If you find this project useful in your research, please use the following BibTeX entry for citation.

```c
```

## Environment & Installation Steps
Installing conda environment

```python
conda env create -f environment.yml
```

## Dataset and Preprocessing 

Download the dataset from [here](https://drive.google.com/drive/u/1/folders/1cI_Hbz0GRoRipJssFVkf9j1suqwK0nmL) and unzip all the three folders.

### Generate data from raw data

```python
python make_price_data.py
```

```python
python make_text_data.py --split train
```
Similarly, make data for `val` and `test` split.

## Run

Execute the following python command to train GPolS: 
```python
python main.py --do_sampling --focal_loss
```