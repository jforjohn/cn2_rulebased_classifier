# cn2_rulebased_classifier
This is a project for implementing and evaluating CN2 rule based classifier.

The folder called cn2_rulebased_classifier} comes with:

  * requirements.txt for downloading possible dependencies (*pip install -r requirements.txt*)
  
  * cn2.cfg configuration file in which you can define the parameters of the algorithm you want to run

When you define what you want to run in the configuration file you just run the MainLauncher.py file.

Concerning the configuration file at the cn2 section:

  * dataset: the name of the dataset in the folder datasets e.g. lymphography (no extension) 
  
  * bins_no: the number of bins that the preprocessing is going to discretize the numerical data

  * min_significance: the minimum significance which acts as a threshold 

  * negate: yes/no, is whether or not to include negations in the rule, default=no

  * disjunctive: yes/no, is whether to create disjunctive or conjuctive rules, default=no

  * train_percentage: (float) the percentage of the train set that the data will be splitted into
  
  * outputs: the directory that the rules will be written to
