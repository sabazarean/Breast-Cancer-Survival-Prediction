# Breast-Cancer-Survival-Prediction
## Installing Requirements
```
pip install -r requirements.txt
```




## Note
Unfortunately, setting  `n_jobs` to more one is not possible due to a bug in pycox library. We mentioned the but in the following issue: ValueError: Need time to have same type as self.durations and proposed the solved the problem in the following pull request: Addressing issue #149 -> change is not to !=.

If your version of pycox is higher that `0.2.3`, and the following line in https://github.com/havakv/pycox/blob/master/pycox/preprocessing/discretization.py#L155 is as follows: `if duration.dtype != self.durations.dtype:` you can safely increase the value of `n_jobs`.

If the mentioned line is as follows:
```
if duration.dtype is not self.durations.dtype:
```
You should set `n_jobs` to 1, or you can open the file `pycox/preprocessing/discretization.py` and change the line to `if duration.dtype != self.durations.dtype:` and increase the value of `n_jobs`
