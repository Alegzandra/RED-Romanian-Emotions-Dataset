![version](https://img.shields.io/badge/version-2-red)

# RED - Romanian Emotions Datasets

We release the second version of the Romanian Emotions Dataset (RED) containing **5449 tweets** annotated in a **multi-label** fashion with the following **7 emotions**: _Anger_ (Furie), _Fear_ (Frică), Joy (_Bucurie_), Sadness (Tristețe),  _Surprise_ (Surpriză), _Trust_ (Încredere) and _Neutral_ (Neutru).

We provide anonymized tweets: we removed usernames and [proper nouns](https://github.com/dumitrescustefan/roner) from the dataset. 

REDv2 is based on REDv1, which is a smaller dataset, single-labeled with 5 emotions (Anger, Fear, Joy, Sadness and Neutral). For compatibility purposes, we keep v1 in the [REDv1](REDv1) folder. If you use REDv1 in your research, please see its [orginal readme](REDv1\readme.md).

# Format

We provide REDv2 split in train/validation/test json files in the [data](data) folder. Each entry has the following format:

```json
{
  "text": <<the anonymized tweet as a UTF8 string",
  "text_id": <<int representing the text id>>,
  "agreed_labels": <<array of 7 ints (0/1) representing the agreed-upon emotion>>,
  "annotator1": <<array of 7 ints (0/1) representing emotions identified by annotator 1>>,
  "annotator2": <<array of 7 ints (0/1) representing emotions identified by annotator 2>>,
  "annotator3": <<array of 7 ints (0/1) representing emotions identified by annotator 3>>,
  "sum_labels": <<array of 7 ints (0-3) representing the sum of the three annotators>>,
  "procentual_labels": <<array of 7 floats [0-1] representing the average sum of the three annotators>> 
}
```

The arrays of 7 values correspond to the following emotions: ``['Tristețe', 'Surpriză', 'Frică', 'Furie', 'Neutru', 'Încredere', 'Bucurie']``.

For example:

```json
{
  "text": "Băi frate... a fost minunat. Bravo, maestre <|PERSON|>! 👏 Și e INCREDIBIL ce poate face vioara aia, a reușit să-mi smulgă o lacrimă și un.. Declar pe proprie răspundere că am avut un eargasm de povestit nepoților și merg la somn complet (audio)satisfăcută.",
  "text_id": 138,
  "agreed_labels": [0, 0, 0, 0, 0, 0, 1],
  "annotator1":    [0, 1, 0, 0, 0, 0, 1],
  "annotator2":    [0, 0, 0, 0, 0, 0, 1],
  "annotator3":    [0, 0, 0, 0, 0, 0, 1],
  "sum_labels":    [0, 1, 0, 0, 0, 0, 3],
  "procentual_labels": [0, 0.33, 0, 0, 0, 0, 1]
}
```

We can see in the anomymized tweet where the person's name was replaced with ``<|PERSON|>`` that all three annotators labeled the text with ``Joy (Bucurie)``, and only one with ``Surprise (Surpriză)``. The ``agreed_labels`` have a 1 only when at least 2 annotators agree on that emotion. 

Given the disagreement between annotators, we can use the dataset in 2 settings:

##### Classification setting

We use the ``agreed_labels`` as the target, and we train a model to predict a True/False for each emotion (multi-label prediction, usually with binary cross entropy). 

##### Regression setting 

The ``procentual_labels`` are the target for a multi-output regression model that attempts to minimize mean squared error across all labels.

# Evaluation

Script and results coming soon in [baseline](baseline).

# Credits

The REDv2 paper is under peer-review and will be published soon.