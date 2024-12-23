# Feature Selection Layer: A Feature Selection Approach for Multi-Class Classification on Neural Networks

> This project aims to create an extension to Feature Selection Layer (FSL) able to assign label-specific feature weights.

In the era of data, deep learning models are becoming increasingly influential in
uncovering patterns and making predictions that directly impact our society. One
of the primary challenges hindering their broader adoption in business and science is
their inherent lack of interpretability. Understanding and justifying how these mod-
els arrive at their decisions is crucial, especially in sensitive domains like healthcare,
biology, and finance. While neural networks excel at discovering hidden patterns
in massive, high-dimensional datasets, they can also inadvertently learn to exploit
biases in the data to achieve high accuracy, leading to biased predictions. This
bias is particularly dangerous when predictions depend on illegitimate factors that
can impact people’s lives. For example, models can mistakenly define gender as
a relevant factor for job performance or race as a factor for criminality, reinforc-
ing social inequalities. In recent years, various post-hoc methods have emerged to
enhance model interpretability by assigning weights to features based on their rel-
evance. These methods typically focus on identifying the most relevant features
for a given prediction, often considering the nuanced relationships between features
and different labels in multi-class problems. To leverage the inner workings of neu-
ral networks for this purpose, embedded methods have been developed to learn
feature importance jointly with the model during the training phase. These meth-
ods generally identify the most relevant features without considering the nuanced
relationships between features and different labels in multi-class problems, unlike
post-hoc techniques. Among these methods is the Feature Selection Layer, which
adds a new layer between the input and the neural network. This layer uses fea-
ture weights that directly affect predictions and are learned jointly with the rest
of the model’s parameters. However, like other embedded methods, the Feature
Selection Layer can only learn general feature weights. To address this limitation,
we propose an extension to Feature Selection Layer designed to capture the specific
relationships between features and individual labels. Our method was evaluated on
both synthetic and real-world datasets supported by multiple evaluation methods
and compared with state-of-the-art post-hoc techniques. It successfully identified
the most relevant features for different labels.

## How to execute
### Define the feature weighting algorithms
On `run_evaluation.py`, add on "selectors_types" list all methods that you want to compare.
All available methods are already imported.
Example: `selectors_types = [MFSLayerV1ReLUSelector, LassoSelectorWrapper, LIMESelectorWrapper, DeepSHAPSelectorWrapper]`
### Setup the general execution config
On `config/general_config.py` is possible to set all framework options including the folder of the dataset that will be used for the evaluation.
### Setup predictors
On `config/predictor_types_config.py` is possible to set the predictors that will be used to evaluate feature selection subsets. 
Current options are: SVC and Neural Network.
### Setup stability metrics
On `config/stability_metrics_config.py` you can define the stability metrics that will be used during the evaluation.
Current options are: Jaccard, Spearman, Pearson and Kuncheva.
### Execute the analysis
With all dependencies (`requirements.txt`) prepared, execute: `python main.py`
