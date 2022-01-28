# RED - Romanian Emotions Dataset v1

This dataset contains tweets in Romanian labelled with the emotions: Anger (Furie), Fear (Frică), Joy (Bucurie), Sadness (Tristețe) and also Neutral (Neutru), split in 3237 tweets for training, 405 tweets for validation and 405 tweets for testing. 

| Class Name | No. of labelled tweets |
| ------- | --- | 
| Anger | 807 | 
| Fear | 778 |
| Joy | 876 |
| Sadness | 781 |
| Neutral | 805 |

To protect confidentiality of Twitter users, we removed usernames and also proper names from this dataset.  

# Create an emotion detection model

To try a BERT-based emotion detecton model trained on RED dataset you can train your own model using train_bert_model.py and the train/test/val split provided in the RED data folder.

# Credits

If you use this dataset in your reasearch, please cite:  

```text
A. Ciobotaru, L.P. Dinu, RED: A Novel Dataset for Romanian Emotion Detection from Tweets, Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2021).
```
and in BibTex format: 

```bash
@inproceedings{RED,
    title = " RED: A Novel Dataset for Romanian Emotion Detection from Tweets",
    author = "Ciobotaru, Alexandra  and Dinu, Liviu P.",
    booktitle = "Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2021)",
    month = sep,
    year = "2021",
    address = "Varna, Bulgaria",
    publisher = "INCOMA Ltd.",
    url = "https://ranlp.org/ranlp2021/proceedings.pdf",
    doi = "https://doi.org/10.26615/978-954-452-072-4_035",
    pages = "296-305",
    abstract = "In Romanian language there are some resources for automatic text comprehension, but for Emotion Detection, not lexicon-based, there are none. To cover this gap, we extracted data from Twitter and created the first dataset containing tweets annotated with five types of emotions: joy, fear, sadness, anger and neutral, with the intent of being used for opinion mining and analysis tasks. In this article we present some features of our novel dataset, and create a benchmark to achieve the first supervised machine learning model for automatic Emotion Detection in Romanian short texts. We investigate the performance of four classical machine learning models: Multinomial Naive Bayes, Logistic  Regression, Support Vector Classification and Linear Support Vector Classification. We also investigate more modern approaches like fastText, which makes use of subword information. Lastly, we finetune the Romanian BERT for text classification and our experiments show that the BERTbased model has the best performance for the task of Emotion Detection from Romanian tweets.",
}
```
