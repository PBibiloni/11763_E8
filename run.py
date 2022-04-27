import logging
import os

from skimage.io import imread
import pandas as pd

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def errors(img_gt, img_segmentation):
    # Task 1.1: Find true positives/negatives, false positives/negatives.
    log.info('Task 1.1: Find true positives/negatives, false positives/negatives.')
    img_gt = img_gt.astype('bool')
    if img_segmentation.ndim == 3:
        img_segmentation = img_segmentation.sum(axis=2)
    img_segmentation = img_segmentation.astype('bool')
    # YOUR CODE HERE
    # ...
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    # ...

    # Return
    return true_positives, true_negatives, false_positives, false_negatives


def performance_metrics(img_gt, img_segmentation):
    log.info('Task 1.2: Compute sensitivity, specificity, f1_score.')
    # YOUR CODE HERE
    # ...
    sensitivity = 0
    specificity = 0
    f1_score = 0
    # ...

    # Return
    return sensitivity, specificity, f1_score


def compute_all_measures(gt_method_name, name_first_method, name_second_method):
    images_gt = [_load_image(gt_method_name, idx) for idx in range(1, 21)]
    images_first_method = [_load_image(name_first_method, idx) for idx in range(1, 21)]
    images_second_method = [_load_image(name_second_method, idx) for idx in range(1, 21)]
    log.info('Task 2.1: Compute ONE performance measure to compare methods with.')
    # YOUR CODE HERE
    # ...
    list_indicators_first_method = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    list_indicators_second_method = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # ...

    # Return
    return list_indicators_first_method, list_indicators_second_method


def hypothesis_testing(list_indicators_first_method, list_indicators_second_method):
    log.info('Task 2.2: Perform hypothesis testing.')
    series_A = pd.Series(list_indicators_first_method)
    series_B = pd.Series(list_indicators_second_method)
    # Import some interesting hypothesis tests
    from scipy.stats import ttest_1samp, ttest_ind, ttest_rel, wilcoxon, bartlett, kstest, shapiro
    # YOUR CODE HERE
    # ...
    pvalue = 0
    first_method_is_statistically_superior = False
    # ...

    # Return
    return pvalue, first_method_is_statistically_superior


def _load_image(method, idx):
    path = f'data/DB_EyeFundus_DRIVE/Dataset/{method}/'
    fname = [n for n in os.listdir(path) if n.startswith(f'{idx:02d}')][0]
    return imread(f'{path}{fname}')


AVAILABLE_METHODS = [
    # 'Originals',
    'Ground Truth - Vessel Segmentation 1',
    'Ground Truth - Vessel Segmentation 2',
    'Results - 1989 Chaud',
    'Results - 1999 Perez',
    'Results - 2001 Zana',
    'Results - 2003 Jiang',
    'Results - 2004 Niemeijer',
    'Results - 2004 Staal',
    'Results - 2008 Soares',
    'Results - 2014 Chowdhury',
    'Results - 2015 Chowdhury',
]

if __name__ == '__main__':
    # Load data
    gt = _load_image('Ground Truth - Vessel Segmentation 1', 1)
    segmentation = _load_image('Results - 2015 Chowdhury', 1)

    log.info('Task 1.1: Find true positives/negatives, false positives/negatives.')
    tp, tn, fp, fn = errors(
        img_gt=gt,
        img_segmentation=segmentation)
    log.info(f'Errors: TP={tp:d}, TN={tn:d}, FP={fp:d}, FN={fn:d}.')

    log.info('Task 1.2: Compute sensitivity, specificity, f1_score.')
    sen, spe, f1 = performance_metrics(
        img_gt=gt,
        img_segmentation=segmentation)
    log.info(f'Errors: Sensitivity={sen:.02%}, Specificity={spe:.02%}, F1 score={f1:.02%}.')

    # Evaluate task 2
    log.info('Task 2.2: Perform hypothesis testing.')
    for first_method in AVAILABLE_METHODS:
        for second_method in AVAILABLE_METHODS:
            if first_method != second_method:
                first_measures, second_measures = compute_all_measures(
                    gt_method_name='Ground Truth - Vessel Segmentation 1',
                    name_first_method=first_method,
                    name_second_method=second_method)
                pvalue, first_is_superior = hypothesis_testing(first_measures, second_measures)
                log.info(f'Method {first_method} (mean {sum(first_measures)/20:.02%}) is statistically superior to method {second_method} (mean {sum(second_measures)/20:.02%})? {first_is_superior} (pvalue={pvalue:.03f}).')
