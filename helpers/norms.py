###################################################################################################
# Copyright (c) 2022 Birgit Hillebrecht
#
# To cite this code in publications, please use
#       B. Hillebrecht and B. Unger : "Certified machine learning: Rigorous a posteriori error bounds for PDE defined PINNs", arxiV preprint available
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
###################################################################################################

from helpers.l2_norm import compute_l2_norm, compute_l2_norm_from_contributions, compute_l2_single_point_norm_contribution

USED_NORM = compute_l2_norm
USED_SINGLE_POINT_NORM_CONTRIBUTION = compute_l2_single_point_norm_contribution
USED_CONTRIBUTION_TO_NORM_FUNCTION = compute_l2_norm_from_contributions

def norm(f, x, outdim=None):
    """
    Returns norm used for this example

    param np.array or tf.tensor f : array over which the L2 norm shall be computed
    param np.array x : underlying space grid 
    param int outdim : number of output dimesnions relevant for the L2 norm. Consider f[:, :outdim] for L2 norm only
    """
    global USED_NORM
    return USED_NORM(f, x, outdim)

def single_point_norm_contribution(f, outdim= None):
    """
    Returns contribution of a single point to the norm (required for domain average computations)
    """
    global USED_SINGLE_POINT_NORM_CONTRIBUTION
    return USED_SINGLE_POINT_NORM_CONTRIBUTION(f, outdim)

def norm_from_contributions(single_point_contribs, x):
    """
    Returns function used to compute the total norm from single point contributions 
    """
    global USED_CONTRIBUTION_TO_NORM_FUNCTION
    return USED_CONTRIBUTION_TO_NORM_FUNCTION(single_point_contribs, x)

def set_current_norm(new_norm, contrib, joinfunction):
    """
    Extract key performance indicators for a posteriori error estimation. KPIs are then stored in a KPI file to be persistently accessible for other evaluation steps 

    :param function new_norm: returns three doubles for two integral computations with different spacings and the thereof derived integration error (I1,I2,E). Requires two (optionally three) arguments: function to be integrated as numpy array of function values (not squared yet), array of equally spaced space-grid values, optionally the number of relevant output dimensions. 
    """
    global USED_NORM, USED_SINGLE_POINT_NORM_CONTRIBUTION, USED_CONTRIBUTION_TO_NORM_FUNCTION
    USED_NORM = new_norm
    USED_SINGLE_POINT_NORM_CONTRIBUTION = contrib
    USED_CONTRIBUTION_TO_NORM_FUNCTION = joinfunction
    return

