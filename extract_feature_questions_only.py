"""
This is simply a dummy file for creating comparable files for the singular feature detectors.
What happens is that the files with the feature detectors are checked, and for each question with the feature,
its index is retrieved and the values for this and the unprocessed are stored in a new DataFrame, which is then
saved to file.
"""
import constants as const
from pandas import DataFrame
from train_classifier import create_and_save_model, create_singular_feature_detector_model

FILE_ENDING = ".csv"
FILENAME_START = "training_data_10000_unprocessed"
NEW_PATH = "." + const.SEPARATOR + "extraction_sets" + const.SEPARATOR


def __get_filename(path, filename):
    return path + FILENAME_START + "_" + filename.strip() + FILE_ENDING


def __extract_single_features(feature):
    """
    Creates two files (one with unprocessed, and one with singular feature) containing questions that has the feature

    Arguments:
        feature: The feature(s) to look for

    Returns:
         tuple (pandas.DataFrame, pandas.DataFrame): Tuple that contains the dataframe with
          updated unprocessed questions (those that contains the given feature), and the
          other dataframe that has the features added to its question text
    """
    up_name = "UP_" + feature.strip()
    new_index = old_index = 0
    path = const.FILEPATH_TRAINING_DATA + FILENAME_START + FILE_ENDING
    unprocessed_df = DataFrame.from_csv(path)
    feature_df = DataFrame.from_csv(__get_filename(const.FILEPATH_FEATURE_DETECTOR, feature))
    new_up_dataframe = DataFrame(columns=unprocessed_df.columns.values)
    new_feat_dataframe = DataFrame(columns=unprocessed_df.columns.values)
    for question in feature_df[const.QUESTION_TEXT_KEY]:
        if feature in question:
            new_feat_dataframe.loc[new_index] = feature_df.loc[old_index].copy()
            new_up_dataframe.loc[new_index] = unprocessed_df.loc[old_index].copy()
            new_index += 1
        old_index += 1
    new_up_dataframe.to_csv(__get_filename(NEW_PATH, up_name), encoding='utf-8')
    new_feat_dataframe.to_csv(__get_filename(NEW_PATH, feature), encoding='utf-8')


def __extract_multiple_features(feature1, feature2, filename):
    """
    Creates two files (one with unprocessed, and one with singular feature) containing questions that has the features

    Arguments:
        feature1: The first feature to look for
        feature2: The second feature to look for
        filename (str): File containing the features

    """
    new_index = old_index = 0
    up_name = "UP_" + filename.strip()
    path = const.FILEPATH_TRAINING_DATA + FILENAME_START + FILE_ENDING
    unprocessed_df = DataFrame.from_csv(path)
    feature_df = DataFrame.from_csv(__get_filename(const.FILEPATH_FEATURE_DETECTOR, filename))
    new_up_dataframe = DataFrame(columns=unprocessed_df.columns.values)
    new_feat_dataframe = DataFrame(columns=unprocessed_df.columns.values)
    for question in feature_df[const.QUESTION_TEXT_KEY]:
        if feature1 in question:
            new_feat_dataframe.loc[new_index] = feature_df.loc[old_index].copy()
            new_up_dataframe.loc[new_index] = unprocessed_df.loc[old_index].copy()
            new_index += 1
        elif feature2 in question:
            new_feat_dataframe.loc[new_index] = feature_df.loc[old_index].copy()
            new_up_dataframe.loc[new_index] = unprocessed_df.loc[old_index].copy()
            new_index += 1
        old_index += 1
    new_up_dataframe.to_csv(__get_filename(NEW_PATH, up_name), encoding='utf-8')
    new_feat_dataframe.to_csv(__get_filename(NEW_PATH, filename), encoding='utf-8')


def __create_new_classifier_model(filename, use_sgd_settings=False):
    """
    Creates a new classifier model based on the data in the passed ```dataframe```

    Arguments:
        filename (str): path + Filename for the model
        use_sgd_settings (bool): If True, run exhaustive SGD search. If False, use exhaustive SVC

    Returns:
         model (sklearn.model_selection._search.GridSearchCV): The created classifier model

    """
    dataframe = DataFrame.from_csv(__get_filename(NEW_PATH, filename), encoding='utf-8')
    print("Retrieving questions and classification labels...")
    training_data = dataframe[const.QUESTION_TEXT_KEY].copy()
    class_labels = dataframe[const.CLASS_LABEL_KEY].copy()
    print("Starting training of model")
    file = NEW_PATH + FILENAME_START + "_UP_" + filename + ".pkl"
    model = create_and_save_model(training_data, class_labels, file, predict_proba=True,
                                  test_size=float(0.2), random_state=0, print_results=True,
                                  use_sgd_settings=use_sgd_settings)
    __create_new_singular_feature_model(filename, model)


def __create_new_singular_feature_model(filename, model):
    """
    Crate a new model based on the singular feature (only for questions containing it)

    Arguments:
        filename (str): Name of file
        model: Unprocessed data set model

    """
    if model is not None:
        pipeline_svm = model.best_estimator_
        # set up the parameter values
        param_svm = [
            {
                'clf__C': [model.best_params_['clf__C']],
                'clf__kernel': [model.best_params_['clf__kernel']],
            },
        ]
        # check if gamma is a part of the parameters
        if model.best_params_.get('clf__gamma') is not None:
            param_svm[0]['clf__gamma'] = [model.best_params_.get('clf__gamma')]
    dataframe = DataFrame.from_csv(__get_filename(NEW_PATH, filename), encoding='utf-8')
    print("Retrieving questions and classification labels...")
    training_data = dataframe[const.QUESTION_TEXT_KEY].copy()
    class_labels = dataframe[const.CLASS_LABEL_KEY].copy()
    print("Starting training of model")
    filename = NEW_PATH + FILENAME_START + "_" + filename + ".pkl"
    create_singular_feature_detector_model(pipeline_svm, param_svm, filename, training_data, class_labels,
                                           test_size=float(0.2), random_state=0)

if True:
    # extract the features which has only one feature defined
    __extract_single_features(const.QUESTION_HAS_HEXADECIMAL_KEY)
    __extract_single_features(const.QUESTION_HAS_NUMERIC_KEY)
    __extract_single_features(const.QUESTION_HAS_LINKS_KEY)
    __extract_single_features(const.QUESTION_HAS_CODEBLOCK_KEY)
    # extract those that have multiple
    __file = "has_tags"
    __extract_multiple_features(const.QUESTION_HAS_ATTACHED_TAG_KEY, const.QUESTION_HAS_EXTERNAL_TAG_KEY, __file)
    __file = const.QUESTION_HAS_HOMEWORK_KEY
    __extract_multiple_features(const.QUESTION_HAS_HOMEWORK_KEY, const.QUESTION_HAS_ASSIGNMENT_KEY, __file)


__create_new_classifier_model(const.QUESTION_HAS_HEXADECIMAL_KEY)
__create_new_classifier_model(const.QUESTION_HAS_NUMERIC_KEY)
__create_new_classifier_model(const.QUESTION_HAS_LINKS_KEY)
__create_new_classifier_model(const.QUESTION_HAS_CODEBLOCK_KEY)
__file = "has_tags"
__create_new_classifier_model(__file)
__file = const.QUESTION_HAS_HOMEWORK_KEY
__create_new_classifier_model(__file)
