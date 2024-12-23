import torch


# Configurations related to training
EPOCHS = 100
BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.001
USE_SGD_OPTIMIZER = False
USE_MODEL_REGULARIZATION = True
SHOULD_LOG_WHILE_TRAINING = False
SHOULD_VALIDATE_WHILE_TRAINING = False
SHOULD_USE_SIMPLER_MODEL = False
REGULARIZATION_LAMBDA = 0.001

# General evaluation
SHOULD_CALCULATE_METRICS_BY_LABEL = True

# Selectors config
DEEP_SHAP_K = 1000
LIME_K = 1000
RELIEFF_K = 1000

# Configurations related to dataset
DATASET_FILE = "synthetic/synth_3classes_3000samples_300features_30informative_15informativeperclass.csv"
DATA_TYPE = torch.float32
CLASS_TYPE = torch.long
TEST_SIZE = 0.2
K_FOLD = 11
K_FOLD_REPEAT = 1
RANDOM_STATE = 42
SHOULD_SCALE = True
SHOULD_STANDARDIZE = True
DATASETS_PATH = "./datasets/"
SYNTHETIC_DATASETS_PATH = './datasets/synthetic'
INFORMATIVE_FEATURE_PREFIX = "informative_"
INFORMATIVE_FEATURE_PER_LABEL_PREFIX = "informative_for_labels_"

# Setup output folder
OUTPUT_PATH = "result-synthC"
TEMP_OUTPUT_PATH = ".temp"
STABILITY_OUTPUT_SUB_PATH = "stability"
OCCLUSION_OUTPUT_SUB_PATH = "occlusion"
INFORMATIVE_FEATURES_OUTPUT_SUB_PATH = "infomative_features"
EXECUTION_TIME_OUTPUT_SUB_PATH = "execution_time"
PREDICTOR_PERFORMANCE_OUTPUT_SUB_PATH  = "predictor"
SELECTOR_PREDICTION_PERFORMANCE_OUTPUT_SUB_PATH = "selector-prediction"
RAW_SELECTION_SUBPATH = "raw-selection"

# Configuration related to informative features
INFORMATIVE_FEATURES_STEP_ON_CHART = 20

# Configurations related to predictor
PREDICTOR_EPOCHS = 25
PREDICTOR_EXECUTIONS = 11
PREDICTOR_INITIAL_STEP = 1 
PREDICTOR_INITIAL_END = 30
PREDICTOR_STEP = 1
PREDICTOR_LIMIT = None 
PREDICTOR_STEP_ON_CHART = 20
PREDICTOR_SHOULD_CREATE_INDIVIDUAL_CHARTS_FOR_EACH_SELECTION_SIZE = False

# Configuration for Occlusion
SHOULD_SAVE_OCCLUSION_VIDEO = False
OCCLUSION_INITIAL_STEP = 1 
OCCLUSION_INITIAL_END = 30
OCCLUSION_STEP = 1 
OCCLUSION_LIMIT = None 
OCCLUSION_STEP_ON_CHART = 20
OCCLUSION_BY_LOSS = True

# Configuration for informative features charts
MAX_INFORMATIVE_FEATURES_CHART_RANGE = 100
FEATURES_TO_DISPLAY_PLUS_INFORMATIVE_ON_HEADMAP = 5
FEATURES_TO_DISPLAY_ON_GENERAL_HEATMAP = 30
CREATE_HEATMAP_BASED_ON_INFORMATIVE_FEATURES = False

# Configuration for stability
STABILITY_LIMIT = None
STABILITY_INITIAL_STEP = 1 
STABILITY_INITIAL_END = 30 
STABILITY_STEP = 1 
STABILITY_STEP_ON_EVOLUTION_CHART = 20
MAX_FEATURES_TO_DISPLAY_ON_SELECTION_STABILITY_CHART = 5000
STABILITY_SHOULD_CREATE_INDIVIDUAL_CHARTS_FOR_EACH_SELECTION_SIZE = False