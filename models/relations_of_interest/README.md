# Running training and evaluation only for relations of interest
To reproduce results of Section 5.2 of our paper

## 1. RE-GCN
* Run the code provided in ```relations_of_interest/roi-RE-GCN```
* This makes sure that for computing the loss during training and for evaluation, only predictions for the relations of interest are made.
* Follow the same instructions as for normal RE GCN

## 2. Timetraveler
* Run the code provided in ```relations_of_interest/roi-timetraveler```
* This makes sure that for computing the loss during training and for evaluation, only predictions for the relations of interest are made.
* Follow the same instructions as for normal RE GCN

## 3. TLogic
* specify the relations of interest in this arg: ```parser.add_argument("--rels_of_interest", '-roi', default=[-1], type=list)```.
* It is set to -1 if all relations should be used
* For our experiments the relations of interest are: ```[4,8,23,27 ]```
* 
## 4. Static Baseline
* in ```./models/static_baseline/engine/engine.py``` set ```MASK= True```.
* This leads to training and evaluation only being done for triples with the relations of interest, specified by mask.
