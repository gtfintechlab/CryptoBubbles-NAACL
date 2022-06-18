# Cryptocurrency Bubble Detection: A New Stock Market Dataset, Financial Task & Hyperbolic Models

This is an official pytorch implementation of our NAACL 2022 paper Cryptocoin Bubble Detection: A New Dataset, Task & Hyperbolic Models. In this repository, we provide PyTorch code for training our proposed MBHN model. We also provide scripts to develop data from the provided raw data. 

If you find this project useful in your research, please use the following BibTeX entry for citation.

```c
@inproceedings{sawhney-etal-2022-cryoto,
    title = "Cryptocurrency Bubble Detection: A New Stock Market Dataset, Financial
Task & Hyperbolic Models",
    author = "Sawhney, Ramit  and
      Agarwal, Shivam  and
      Mittal, Vivek and
      Rosso, Paolo and
      Nanda, Vikram and
      Chava, Sudheer"
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2022",
    address = "Seattle, Washington",
    publisher = "Association for Computational Linguistics",
    abstract = "The rapid spread of information over social media influences quantitative trading and investments. 
The growing popularity of speculative trading of highly volatile assets such as cryptocurrencies and meme stocks presents a fresh challenge in the financial realm. 
Investigating such bubbles - periods of sudden anomalous behavior of markets are critical in better understanding investor behavior and market dynamics.
However, high volatility coupled with massive volumes of chaotic social media texts, especially for underexplored assets like cryptocoins pose a challenge to existing methods. 
Taking the first step towards NLP for cryptocoins, we present and publicly release CryptoBubbles, a novel multi-span identification task for bubble detection, and a dataset of more than 400 cryptocoins from 9 exchanges over five years spanning over two million tweets.
Further, we develop a set of sequence-to-sequence hyperbolic models suited to this multi-span identification task based on the power-law dynamics of cryptocurrencies and user behavior on social media.
We further test the effectiveness of our models under zero-shot settings on a test set of Reddit posts pertaining to 29 ``meme stocks'', which see an increase in trade volume due to social media hype. 
Through quantitative, qualitative, and zero-shot analyses on Reddit and Twitter spanning cryptocoins and meme-stocks, we show the practical applicability of CryptoBubbles and hyperbolic models.",
}
```

## Environment & Installation Steps
Installing conda environment

```python
conda env create -f environment.yml
```

## Dataset and Preprocessing 

Download the dataset from [here](https://zenodo.org/record/6556673#.Yq4UIOxBy3I) and unzip all the three folders.

### Generate data from raw data

```python
python make_price_data.py
```

```python
python make_text_data.py --split train
```
Similarly, make data for `val` and `test` split.

## Run

Execute the following python command to train MBHN: 
```python
python main.py --do_sampling --focal_loss
```

