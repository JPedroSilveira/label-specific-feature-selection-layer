import config.general_config as general_config
from util.file_util import create_folder_if_not_exists


def create_output_files():
    '''
    Create all necessary folders to persist all outputs
    '''
    create_folder_if_not_exists(general_config.OUTPUT_PATH)
    create_folder_if_not_exists(general_config.TEMP_OUTPUT_PATH)
    create_folder_if_not_exists(f'{general_config.OUTPUT_PATH}/{general_config.STABILITY_OUTPUT_SUB_PATH}')
    create_folder_if_not_exists(f'{general_config.OUTPUT_PATH}/{general_config.OCCLUSION_OUTPUT_SUB_PATH}')
    create_folder_if_not_exists(f'{general_config.OUTPUT_PATH}/{general_config.INFORMATIVE_FEATURES_OUTPUT_SUB_PATH}')
    create_folder_if_not_exists(f'{general_config.OUTPUT_PATH}/{general_config.EXECUTION_TIME_OUTPUT_SUB_PATH}')
    create_folder_if_not_exists(f'{general_config.OUTPUT_PATH}/{general_config.PREDICTOR_PERFORMANCE_OUTPUT_SUB_PATH}')
    create_folder_if_not_exists(f'{general_config.OUTPUT_PATH}/{general_config.SELECTOR_PREDICTION_PERFORMANCE_OUTPUT_SUB_PATH}')
    create_folder_if_not_exists(f'{general_config.OUTPUT_PATH}/{general_config.RAW_SELECTION_SUBPATH}')