# ML classification model

Model learns to predict a class on texts meaning. Neural Net's architecture is stacked CNN layers. Texts are transformed into one-hot matrices.
1. Check requirements.txt
2. Download data from https://www.kaggle.com/zynicide/wine-reviews. Other data source can be used, if provided in a Pandas dataframe with columns "description" for texts and "country" for labels.
3. Run dataset_analysis.ipynb to see if it loads correctly and get handy dataset info.
4. Get you GPU ready, the model is a little bit on a heavy side. CPU is not advisable. M1 takes more than an hour to train.
5. Run multiclass_classification.py to train the model.
6. If you have target data to get prediction, put it in reviews_for_print in main(). Otherwise random texts from the dataset will be used for it.




