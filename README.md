# ml_flood

## ESoWC 2019 - MATEHIW // MAchine learning TEchniques for High-Impact Weather

**Goal:** A comparison study between different ML algorithms on forecasting flood events using open datasets from ECMWF/Copernicus.

**Team:** [@lkugler](https://github.com/lkugler), [@seblehner](https://github.com/seblehner)

### The project is work in progress and will be frequently updated!

## Table of Contents


* [Project description](#Project-description)
* [Dependencies and Setup](#Dependencies-and-Setup)
* [Data description](#Data-description)
* [ML techniques](#ML-techniques)
* [Acknowledgments](#Acknowledgments)


### Project description

We plan to investigate various machine learning (ML) techniques for predicting floods. The main goal is a comparative study of some of the most promising ML methods on this proposed subject. As a side goal, the open source development approach via github will result in a nice basis for further work.

ERA5 data will be used as predictor to model either the probability of exceeding some threshold in river discharge by the GloFAS reanalysis or to predict the severeness of the event given by the ECMWF’s severe event catalogue. We plan to investigate the impact of different meteorological variables, starting with direct precipitation output and combinations of thermodynamic and dynamic variables.
Additionally, the results can be compared with GloFAS forecast reruns as well. Thereby, the benefits and/or drawbacks of using ML techniques instead of coupled complex models can be explored.

Our projected workflow can be seen below:

![img](https://raw.githubusercontent.com/seblehner/seblehner.github.io/master/images/MATEHIW_flowchart.png)

### Dependencies and Setup
**Some modules require Python 3!** Dependencies can be found in the **environment.yml** file. Download the repository, move it to any path you wish for. You can either install all packages by hand, or you can use `
```sh
conda env create -f environment.yml
```
inside the
```sh
/ml_flood/
```
folder for a one-step installation of all dependencies. When installed, a new environment named **ml_flood** is created. Remember to use
```sh
source activate ml_flood
```
before executing any files.


### Data description
We use ERA5 Reanalysis and GloFAS Reanalysis and forecast rerun data. A detailed description can be found in the notebook [003_data_overview](https://github.com/esowc/ml_flood/blob/dev/docs/003_data_overview.ipynb).



### ML techniques

work in progress

### Acknowledgments
We acknowledge the support of ECMWF for bringing this project to life!
