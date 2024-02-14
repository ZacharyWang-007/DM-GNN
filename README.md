# Pytorch implementation of Dual-stream multi-dependency graph neural network enables precise cancer survival analysis (DM-GNN)

This is for the under review "paper Dual-stream multi-dependency graph neural network enables precise cancer survival analysis". 

 ## Requirements
 ### Installation
Please install pytorch version >=1.2

 ### Dataset Preparation
 Please download the official [TCGA datasets](https://www.cancer.gov/ccg/research/genome-sequencing/tcga) of BRCA, BLCA, GBMLGG, LUAD, and UCEC. 
 For more details on pre-processing, please refer to [CLAM](https://github.com/mahmoodlab/CLAM).
 
 ## Model training and testing
 before training and testing, please update configs. Generally, we train the model with one 24 GB memory GPU. You can adjust the 'num_instances_maximum' to sample the number of instances in accordance with your GPU power.  
 ~~~~~~~~~~~~~~~~~~
   python main_blca.py/main_brca.py/main_gbmlgg.py/main_luad.py/main_ucec.py
 ~~~~~~~~~~~~~~~~~~

Given that this work is still under review, partial of the coded regarding the neural network hasn't been uploaded. 

## Contact
If you have any questions, please don't hesitate to contact us. E-mail: [zhikang.wang@monash.edu](zhikang.wang@monash.edu) 

