# LexiconNER
This is the implementation of "[Distantly Supervised Named Entity Recognition using Positive-Unlabeled
Learning](https://www.aclweb.org/anthology/P19-1231)" published at ACL 2019. The highlight of this work is it performs NER using only entity dictionaries without any labeled data.

By the way, we recently publish our another work related to Chinese NER. It designs to augment Chinese NER with lexicons. The highlight of this work is that it has high computational efficiency and at the same time, achieves comparative or better performance over existing methods. You can access the source code of that work and a hyper-link of its associated paper at [LexiconAugmentedNER](https://github.com/v-mipeng/LexiconAugmentedNER).

### Set up and run
Download glove.6B.100d.txt
### Environment
pytorch 1.1.0
python 3.6.4
cuda 8.0
### Instructions for running code
#### Phrase one \<train bnPU model\>
**Train**
Print parameters
`run python feature_pu_model.py  --h`
```html
optional arguments:
  -h, --help            show this help message and exit
  --lr LR               learning rate
  --beta BETA           beta of pu learning (default 0.0)
  --gamma GAMMA         gamma of pu learning (default 1.0)
  --drop_out DROP_OUT   dropout rate
  --m M                 class balance rate
  --flag FLAG           entity type (PER/LOC/ORG/MISC)
  --dataset DATASET     name of the dataset
  --batch_size BATCH_SIZE
                    	batch size for training and testing
  --print_time PRINT_TIME
                    	epochs for printing result
  --pert PERT           percentage of data use for training
  --type TYPE           pu learning type (bnpu/bpu/upu)
```
e.g.)
Train on PER type of conll2003 dataset:
`python feature_pu_model.py --dataset conll2003 --flag PER`
** Evaluating**
```html
python feature_pu_model_evl.py --model saved_model/bnpu_conll2003_PER_lr_0.0001_prior_0.3_beta_0.0_gamma_1.0_percent_1.0 --flag PER --dataset conll2003 --output 1
```
replace the model name from the training
```html
python final_evl.py 
```
Get the final result on all the entity type. Remember to revise the filenames to be the output file name of evaluating.

#### Phrase two \<train adaPU model\>
**dictionary generation**
`run python ada_dict_generation.py -h`
```html
optional arguments:
  -h, --help            show this help message and exit
  --beta BETA           learning rate
  --gamma GAMMA         gamma of pu learning (default 1.0)
  --drop_out DROP_OUT   dropout rate
  --m M                 class balance rate
  --flag FLAG           entity type (PER/LOC/ORG/MISC)
  --dataset DATASET     name of the dataset
  --lr LR               learning rate
  --batch_size BATCH_SIZE
                        batch size for training and testing
  --iter ITER           iteration time
  --unlabeled UNLABELED
                        use unlabeled data or not
  --pert PERT           percentage of data use for training
  --model MODEL         saved model name
```
e.g.)
`python ada_dict_generation.py --model saved_model/bnpu_conll2003_PER_lr_0.0001_prior_0.3_beta_0.0_gamma_1.0_percent_1.0 --flag PER --iter 1`
**adaptive training**
`run python adaptivepumodel.py -h `
````html
optional arguments:
  -h, --help            show this help message and exit
  --beta BETA           beta of pu learning (default 0.0)
  --gamma GAMMA         gamma of pu learning (default 1.0)
  --drop_out DROP_OUT   dropout rate
  --m M                 class balance rate
  --p P                 estimate value of prior
  --flag FLAG           entity type (PER/LOC/ORG/MISC)
  --dataset DATASET     name of the dataset
  --lr LR               learning rate
  --batch_size BATCH_SIZE
                        batch size for training and testing
  --output OUTPUT       write the test result, set 1 for writing result to
                        file
  --model MODEL         saved model name
  --iter ITER           iteration time
````
e.g.)
`python adaptive\_pu\_model.py --model saved\_model/bnpu\_conll2003\_PER\_lr\_0.0001\_prior\_0.3\_beta\_0.0\_gamma\_1.0\_percent\_1.0 --flag PER --iter 1`
Replace saved model names and iteration times when doing adaptive learning. And in the same iteration the iter number in dictionary generation and adaptive learning should be same.

### Cite

Please cite our ACL 2019 paper:

```
@article{peng2019distantly,
  title={Distantly Supervised Named Entity Recognition using Positive-Unlabeled Learning},
  author={Peng, Minlong and Xing, Xiaoyu and Zhang, Qi and Fu, Jinlan and Huang, Xuanjing},
  journal={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2019}
}
```
