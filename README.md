# How to get preferred encoding in python
We used time-to-event survival prediction models including `Nnet-survival`, `DeepHit`, `Cox-Time` and `MLTR` from Pycox package :https://github.com/havakv/pycox to predict breast cancer survival data. we developed the models with hyperparameter grid search and by labaling breast cancer data in four molecular subtypes, the survival probabilities were predicted in each subtypes. then we figure the Kaplan-Meyer curve for each subtype in 10 time intervals to analyze the patterns of survival probabilities in each subtype.

## Background: 
> Breast cancer (BC) survival prediction can be a helpful tool for identifying important factors, selecting the effective treatment, and reducing mortality rates. This study aims to predict the survival probability of BC patients in different molecular subtypes defined by estrogen receptor (ER), progesterone receptor (PR), and human epidermal growth factor receptor 2 (HER2) over 30 years of follow-up.
## Materials and Methods: 
> This is a retrospective cohort of 3580 patients diagnosed with invasive BC from 1991 to 2021 in the Cancer Research Center of Shahid Beheshti University of Medical Science. The dataset contained 20 predictor variables and two dependent variables, which referred to the survival status of patients and the time patients survived from diagnosis. The feature selection was conducted using the random forest algorithm to determine the significant prognostic factors. Time-to-event deep-learning-based models, including Nnet-survival, DeepHit, DeepSurve, NMLTR and Cox-time, were developed by a grid search first with total variables and second with the most important variables resulting from feature selection. C-index and IBS were used as the performance metrics to find the best-performing model. The dataset was clustered based on the molecular receptor status of patients (i.e., luminal A, luminal B, HER2-enriched and triple-negative), and the best-performing prediction model was used to predict the survival probability of patients in each molecular subtype.
## Results: 
> The random forest method identified tumor state, age at diagnosis, and lymph node status as the best subset of variables. All models yielded very close performance, with Nnet-survival (C-index=0.77, IBS=0.13) slightly higher using both total and the three most important variables. Luminal A indicated the highest predicted BC survival probabilities among the four subtypes, whereas triple-negative and HER2-enriched showed the lowest predicted survival probabilities over time. The luminal B subtype followed a similar trend as luminal A for the first five years, after which the predicted survival probability decreased steadily in 10- and 15 years.
## Conclusion: 
> The results of this study can help healthcare providers determine patients' survival probability and prevent unnecessary medical interventions for high-risk patients based on molecular receptor status, especially for HER2-positive patients. Future clinical trials of BC treatments should investigate the response of different molecular subtypes to treatment.

## Installing Requirements
```
pip install -r requirements.txt
```




## Note
Unfortunately, setting  `n_jobs` to more than one is not possible due to a bug in pycox library.
We mentioned the bug in the following issue: ValueError: Need time to have same type as self.durations and proposed a solution to the problem in the following pull request: Addressing issue #149 -> change is not to !=.

If your version of pycox is higher that `0.2.3`, and the following line in https://github.com/havakv/pycox/blob/master/pycox/preprocessing/discretization.py#L155 is as follows: `if duration.dtype != self.durations.dtype:` you can safely increase the value of `n_jobs`.

If the mentioned line is as follows:
```
if duration.dtype is not self.durations.dtype:
```
You should set `n_jobs` to 1, or you can open the file `pycox/preprocessing/discretization.py` and change the line to `if duration.dtype != self.durations.dtype:` and increase the value of `n_jobs`

You also can install the `pycox` in https://github.com/pooya-mohammadi/pycox using the following code:
```commandline
pip install git+https://github.com/pooya-mohammadi/pycox
```

## Data Access

Unfortunately due to the policies and guidelines of the Cancer Research Center of Shahid Beheshti University of Medical Science, data is not allowed for public. for data access request please visit https://crc.sbmu.ac.ir/ 
