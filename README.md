# Investigating Membership Inference Attacks under Data Dependencies
Original code by Bargav Jayaraman: https://github.com/bargavj/EvaluatingDPML

Code modified by: Thomas Humphries, Simon Oya, Matthew Rafuse, and Lindsey Tulloch (ordered alphabetically).

## Obtaining the datasets
All [UCI datasets](http://archive.ics.uci.edu/ml) can be directly downloaded from their corresponding URLs:
- Heart: https://archive.ics.uci.edu/ml/datasets/Heart+Disease
- Census: https://archive.ics.uci.edu/dataset/117/census+income+kdd
- Adult: https://archive.ics.uci.edu/ml/datasets/adult
- Student: https://archive.ics.uci.edu/ml/datasets/Student+Performance (we only take ```student-por.csv``` (Portuguese) because it has the most rows)

The Texas dataset can be obtained by downloading the ```PUDF_base1q2006_tab.txt``` file from: https://www.dshs.texas.gov/THCIC/Hospitals/Download.shtm

The Compas dataset can be obtained by downloading the ```compas-scores-two-years.csv``` file from: https://github.com/propublica/compas-analysis

We note that the exact intermediary files used in our experiments are available upon request (provided that all the terms and conditions of the original data owners are satisfied). However, we have included the code to generate all such intermediary files.

## Running the Code

To run the code locally you will need to install all the dependencies from ```requirements.txt``` 

Specifically: 
matplotlib==3.3.3
scipy==1.5.4
numpy==1.19.4
pandas==1.1.4
lasagne==0.1
mkl==2021.2.0
scikit_learn==0.24.2
tensorflow==2.5.0
tensorflow_privacy==0.5.2
theano==1.0.5
dp_accounting==0.3.0

The experiments can be run using the ```run_all.sh``` script. Note that running all experiments will take a significant amount of time. For a simple test, one should reduce the number of runs and/or the number of $`\epsilon`$ values (which are set in ```config.py```).
