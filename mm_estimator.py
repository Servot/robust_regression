import numpy as np
import rpy2.robjects.packages as rpackages
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri

from typing import Dict, Optional


def get_mm_estimator_coefficients(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
    robustbase = rpackages.importr('robustbase')
    rpy2.robjects.numpy2ri.activate()
    # convert data to r objects
    ro.r.assign("X_train", ro.r.matrix(X_train, nrow=X_train.shape[0], ncol=X_train.shape[1]))
    ro.r.assign("y_train", ro.r.matrix(y_train, nrow=y_train.shape[0], ncol=y_train.shape[1]))

    mm = robustbase.lmrob('y_train ~ X_train')

    try:
        return {
            'MM_ESTIMATOR': mm.rx2('coefficients'),
            'S_ESTIMATOR': mm.rx2('init.S').rx2('coefficients')
        }
    except AttributeError:
        return {
            'MM_ESTIMATOR': mm.rx2('coefficients'),
            'S_ESTIMATOR': None
        }
