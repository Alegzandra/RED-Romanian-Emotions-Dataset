# REDv2 Evaluation 

We test the evaluation script in 2 settings (categorical and regression) with default parameters, and average results over 5 iterations. Results are shown below:

| Model                         	| Setting        	| Hamming Loss 	| Accuracy 	|  MSE  	|
|-------------------------------	|----------------	|:------------:	|:--------:	|:-----:	|
| bert-base-romanian-cased-v1   	| Classification 	|     0.105    	|   0.549  	| 24.30 	|
| bert-base-romanian-cased-v1   	| Regression     	|     0.098    	|   0.543  	| 10.33 	|
| bert-base-romanian-uncased-v1 	| Classification 	|     0.104    	|   0.551  	| 23.95 	|
| bert-base-romanian-uncased-v1 	| Regression     	|     0.097    	|   0.542  	| 10.50 	|
| xlm-roberta                   	| Classification 	|     0.111    	|   0.536  	| 17.22 	|
| xlm-roberta                   	| Regression     	|     0.102    	|   0.546  	| 10.06 	|
| readerbench/RoGPT2-base         | Classification  |               |     |     |
| readerbench/RoGPT2-base         | Regression      |     0.108     |   0.506   | 12.49   |



