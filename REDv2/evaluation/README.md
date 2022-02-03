# REDv2 Evaluation 

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



