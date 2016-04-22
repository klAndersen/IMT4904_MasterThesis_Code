# IMT4904 MasterThesis Pickle Models

These are the models created by using a somewhat exhaustive gridsearch with a given 
 set of parameters. The models have been dumped by using pickle. 
The following is a short  explanation to the files:
- '*_detector': Algorithm used to create model, where '*' is name of model.
- '*_detector_**_': The '**' indicates how model was trained 
(e.g. all=all samples, split=split by using train_test_split)
- Numeric value: Amount of samples per class used to create the model 
(e.g. 10 = 10 good and 10 bad, total 20 samples) 
 