##  Background 
**DDRN: Deep Disentangled Representation Network for Treatment Effect Estimation**

This is a PyTorch implementation of the ***Deep Disentangled Representation Network*** for the task of counterfactual regression, as described in our paper that has been submitted to CIKM 2024 and is currently under review.
## Requirements
Python>=3.6.8 \
PyTorch>=1.10 \
Scikit-uplift>=0.5.1 \
Numpy==1.26.3 \
Scikit-learn==1.3.2



## Try it Out
We provide a demonstration example based on dataset ACIC2016. Specifically,
please first clone the code from https://github.com/gloriousmeng/DDRN.git to your local machine, and execute the following command:

>cd ..\
>python run.py

## Datasets
In the ```DDRN/datasets/``` directory, we provide the IHDP, ACIC 2016 and ACIC public benchmark datasets.
In addition, to facilitate learning and research for our readers, 
we are planning to make the real-wold industrial dataset used in our paper publicly available. Stay tuned for its release.


## Citation
```angular2html
@inproceedings{huimeng2024ddrn,
  title={Deep Disentangled Representation Network for Treatment Effect Estimation},
  author={Hui Meng, Keping Yang, Xuyu Peng, Yifan Shen and Bo Zheng},
  organization={Alibaba Group},
  year={2024}
}
```