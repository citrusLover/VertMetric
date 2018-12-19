
from datetime import datetime
import logging
import numpy as np

"""
File: general.py
Author: Jacob Krantz
Description:
    Misc. functions called for general utility purposes.
"""

def tokenize(txt):
    return txt.strip('\n')

def generate_filename(scope):
    time = datetime.now().strftime('%B-%d_%H:%M:%S_')
    return time + scope + '-score-report.json'

def avg_word_count(lst):
    """
    Computes the average word count of each list element by splitting
        on whitespace.
    Returns:
        float
    """
    return float(np.mean(list(map(lambda s: len(s.split()), lst))))

def verify_data(*lists):
    """
    Throws an error if the generated and/or targets lists are
        incorrect for comparison.
    Returns:
        None
    """
    if len(lists) == 0:
        return 

    try:
        for l in lists[1:]:
            assert isinstance(l, list)
    except AssertionError as err:
        logger = logging.getLogger('vert')
        logger.exception("both 'generated' and 'targets' must be of type 'list'.")
        raise err

    try:
        length = len(l[0])
        for l in lists[1:]:
            assert len(l) == length
    except AssertionError as err:
        logger = logging.getLogger('vert')
        logger.exception("Unequal number of summaries in generated vs target files.")
        raise err

    try:
        assert len(l[0]) > 0
    except AssertionError as err:
        logger = logging.getLogger('vert')
        logger.exception("0 summaries being compared.")
        raise err

def check_data_loaded(generated, targets):
    """
    Ensures two data structures are not empty. Useful for metrics.
        Also calls verify_data.
    """
    if len(generated) != 0 and len(targets) != 0:
        verify_data(generated, targets)
        return
    msg =   '''Generated and target data must be set prior to calling 'score'.
            Either call 'set_generated_and_targets' or 'load_files' '''
    logger = logging.getLogger('vert')
    logger.exception(msg)
    raise UnboundLocalError(msg)

def fmt_rpt_line(l):
    """ format a numerical report line to 3 decimal places. """
    return '{0:.3f}'.format(float(l))
