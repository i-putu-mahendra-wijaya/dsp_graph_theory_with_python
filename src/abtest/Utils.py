import scipy.stats as st
import numpy as np


def get_sample_size(
        mde_rel: float,
        base_std: float = None,
        base_mean: float = None,
        base_prop: float = None,
        is_proportion: bool = True,
        alpha: float = 0.05,
        beta: float = 0.2,
        two_tail: bool = False,
        *args, **kwargs
) -> np.ndarray:
    """

    This function aims to calculate sample size given the :

    mde_rel : MDE relative (%) to the current state.
    base_std : base standard deviation (continuous metrics). Wont be used if proportion set to true.
    base_mean : base mean (continuous metrics). Wont be used if proportion set to true.
    base_prop : base proportion (proportion metrics). Wont be used if proportion set to False.
    is_proportion : whether the metrics is proportion  or continuous . Default True.
    alpha : alpha (probability of false positive error). Default = 0.05.
    beta : beta (probability of false negative error). Default = 0.2.
    two_tail : whether the alpha is two tail or one tail. Default False.



    Example For continuous-metric case  :
    The upcoming campaign are expected to increase the GMV from 2.7 Mio to 3.24 Mio (std = 1.96 Mio), which is 20% increase.
    Alpha is 10% and Beta is 30% while calculation format is one-tail.

    Calculate the sample size by running the function as  :
    get_sample_size(
        mde_rel = 0.2,
        base_std = 1.96e+6,
        base_mean = 2.72e+6,
        is_proportion = False,
        two_tail = False,
        alpha = 0.1,
        beta = 0.3
    )

    Example for proportion-metric case :
    The upcoming campaign are expected to increase app conversion rate from 5% to 6.5% , which is 30% relatively higher.
    Alpha is 10% and Beta is 20% while calculation format is one-tail.

    Calculate the sample size by running the function as  :
    get_sample_size(
        mde_rel = 0.3,
        base_prop = 0.05,
        is_proportion = True,
        alpha = 0.05,
        beta = 0.2,
        two_tail=False
    )
    """

    if is_proportion:  # If the metrics is proportion
        if base_prop > 1:  # No proportion is greater than 1
            raise ValueError('Proportion is bigger than 1')
        else:
            pooled_prop = (base_prop + (base_prop * (1 + mde_rel))) / 2  # Calculate the pooled proportion
            pooled_var = pooled_prop * (1 - pooled_prop)  # Calculate the variance
            d = mde_rel * base_prop  # Calculate the MDE
    else:
        pooled_var = base_std ** 2  # Calculate the variance
        d = mde_rel * base_mean  # Calculate the MDE

    z_a = st.norm.ppf((alpha / (2 if two_tail else 1)))  # Find the z critical given alpha
    z_b = st.norm.ppf(beta)  # Find the z critical given beta

    N_ = (((z_a + z_b) ** 2) * 2 * pooled_var) / (d ** 2)

    return np.floor(N_)