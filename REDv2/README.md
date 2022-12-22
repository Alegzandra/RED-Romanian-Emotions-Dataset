# RED - Romanian Emotions Datasets v2

We release the second version of the Romanian Emotions Dataset (RED) containing **5449 tweets** annotated in a **multi-label** fashion with the following **7 emotions**: _Anger_ (Furie), _Fear_ (Frică), Joy (_Bucurie_), Sadness (Tristețe),  _Surprise_ (Surpriză), _Trust_ (Încredere) and _Neutral_ (Neutru).

We provide anonymized tweets: we removed usernames and [proper nouns](https://github.com/dumitrescustefan/roner) from the dataset. 

REDv2 is based on REDv1, which is a smaller dataset, single-labeled with 5 emotions (Anger, Fear, Joy, Sadness and Neutral). 

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

The arrays of 7 values correspond to the following emotions: ``['Tristețe', 'Surpriză', 'Frică', 'Furie', 'Neutru', 'Încredere', 'Bucurie']`` (``['Sadness', 'Surprise', 'Fear', 'Anger', 'Neutral', 'Trust', 'Joy']``).

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

We test the evaluation script in 2 settings (categorical and regression) with default parameters, and average results over 5 runs with random seed. Results are shown below:

| Model                         	| Setting        	| Hamming Loss 	| Accuracy 	|  MSE  	|
|-------------------------------	|----------------	|:------------:	|:--------:	|:-----:	|
| bert-base-romanian-cased-v1   	| Classification 	|     0.105    	|   0.549  	| 24.30 	|
| bert-base-romanian-cased-v1   	| Regression     	|     0.098    	|   0.543  	| 10.33 	|
| bert-base-romanian-uncased-v1 	| Classification 	|     0.104    	| **0.551**	| 23.95 	|
| bert-base-romanian-uncased-v1 	| Regression     	|   **0.097**  	|   0.542  	| 10.50 	|
| xlm-roberta-base              	| Classification 	|     0.111    	|   0.536  	| 17.22 	|
| xlm-roberta-base               	| Regression     	|     0.102    	|   0.546  	| 10.06 	|
| readerbench/RoGPT2-base         | Classification  |     0.107     |   0.531   | 46.51   |
| readerbench/RoGPT2-base         | Regression      |     0.108     |   0.506   | 12.49   |
| readerbench/RoGPT2-medium       | Classification  |     0.115     |   0.497   | 41.58   |
| readerbench/RoGPT2-medium       | Regression      |     0.104     |   0.511   | 11.11   |

# Credits

```
Ciobotaru, A., Constantinescu, M. V., Dinu, L. P., & Dumitrescu, S. D. (2022). RED v2: Enhancing RED Dataset for Multi-Label Emotion Detection. Proceedings of the 13th Language Resources and Evaluation Conference (LREC 2022), 1392–1399. http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.149.pdf
```

or in bibtex format:

```
@inproceedings{redv2,
  author = "Alexandra Ciobotaru and
               Mihai V. Constantinescu and
               Liviu P. Dinu and
               Stefan Daniel Dumitrescu",
  title = "{RED} v2: {E}nhancing {RED} {D}ataset for {M}ulti-{L}abel {E}motion {D}etection",
  journal = "Proceedings of the 13th Language Resources and Evaluation Conference (LREC 2022)",
  pages = "1392–1399",
  year = "2022",
  address = "Marseille, France",
  publisher = "European Language Resources Association (ELRA)",
  url = "http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.149.pdf",
  language = "English"
}
```
