
# RED - Romanian Emotions Datasets

The Romanian Emotions Datasets are Twitter based datasets annotated with emotions. 

Currently there are **2 releases** available:


|       	|      Link     	| # of tweets 	| # of emotions 	|  Annotation  	| Release Date 	|
|-------	|:-------------:	|:-------------:	|:-------------:	|:------------:	|:------------:	|
| ![version](https://img.shields.io/badge/RED-gray) | [Data & Readme](REDv1) 	| 4047 |       5       	| Single-label 	|  Sep 2021     	|
| ![version](https://img.shields.io/badge/RED-v2-red) | [Data & Readme](REDv2)  | 5449 |       7       	|  Multi-label 	|  Jan 2022 	|

REDv2 is the improved version of REDv1, which is a smaller dataset, single-labeled with 5 emotions (Anger, Fear, Joy, Sadness and Neutral). REDv2 adds Trust and Surprise, bringing the number of annotated emotions to 7, in a multi-label fashion. 

The datasets are available as CSVs and/or JSONs, and are pre-split in train/dev/test. Baselines are provided in their respective folder. 

If you use these datasets in your research/production, kindly cite the appropriate paper, available as bibtext:

### RED:

```bash
@inproceedings{ciobotaru-dinu-2021-red,
    title = "{RED}: A Novel Dataset for {R}omanian Emotion Detection from Tweets",
    author = "Ciobotaru, Alexandra  and
      Dinu, Liviu P.",
    editor = "Mitkov, Ruslan  and
      Angelova, Galia",
    booktitle = "Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2021)",
    month = sep,
    year = "2021",
    address = "Held Online",
    publisher = "INCOMA Ltd.",
    url = "https://aclanthology.org/2021.ranlp-1.34/",
    pages = "291--300",
    abstract = "In Romanian language there are some resources for automatic text comprehension, but for Emotion Detection, not lexicon-based, there are none. To cover this gap, we extracted data from Twitter and created the first dataset containing tweets annotated with five types of emotions: joy, fear, sadness, anger and neutral, with the intent of being used for opinion mining and analysis tasks. In this article we present some features of our novel dataset, and create a benchmark to achieve the first supervised machine learning model for automatic Emotion Detection in Romanian short texts. We investigate the performance of four classical machine learning models: Multinomial Naive Bayes, Logistic Regression, Support Vector Classification and Linear Support Vector Classification. We also investigate more modern approaches like fastText, which makes use of subword information. Lastly, we fine-tune the Romanian BERT for text classification and our experiments show that the BERT-based model has the best performance for the task of Emotion Detection from Romanian tweets. Keywords: Emotion Detection, Twitter, Romanian, Supervised Machine Learning"
}
```

### REDv2

```bash
@inproceedings{ciobotaru-etal-2022-red,
    title = "{RED} v2: Enhancing {RED} Dataset for Multi-Label Emotion Detection",
    author = "Ciobotaru, Alexandra  and
      Constantinescu, Mihai Vlad  and
      Dinu, Liviu P.  and
      Dumitrescu, Stefan",
    editor = "Calzolari, Nicoletta  and
      B{\'e}chet, Fr{\'e}d{\'e}ric  and
      Blache, Philippe  and
      Choukri, Khalid  and
      Cieri, Christopher  and
      Declerck, Thierry  and
      Goggi, Sara  and
      Isahara, Hitoshi  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Mazo, H{\'e}l{\`e}ne  and
      Odijk, Jan  and
      Piperidis, Stelios",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.149/",
    pages = "1392--1399",
    abstract = "RED (Romanian Emotion Dataset) is a machine learning-based resource developed for the automatic detection of emotions in Romanian texts, containing single-label annotated tweets with one of the following emotions: joy, fear, sadness, anger and neutral. In this work, we propose REDv2, an open-source extension of RED by adding two more emotions, trust and surprise, and by widening the annotation schema so that the resulted novel dataset is multi-label. We show the overall reliability of our dataset by computing inter-annotator agreements per tweet using a formula suitable for our annotation setup and we aggregate all annotators' opinions into two variants of ground truth, one suitable for multi-label classification and the other suitable for text regression. We propose strong baselines with two transformer models, the Romanian BERT and the multilingual XLM-Roberta model, in both categorical and regression settings."
}
```
