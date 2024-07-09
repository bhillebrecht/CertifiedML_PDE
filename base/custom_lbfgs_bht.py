###################################################################################################
# Copyright (c) 2024 Birgit Hillebrecht
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
#
# Adapted from https://github.com/yaroslavvb/stuff/blob/master/eager_lbfgs/eager_lbfgs.py
#
###################################################################################################

import tensorflow as tf
import time
import logging

from lbfgs_state import State

# Time tracking functions
global_time_list = []
global_last_time = 0


def reset_time():
    global global_time_list, global_last_time
    global_time_list = []
    global_last_time = time.perf_counter()


def record_time():
    global global_last_time, global_time_list
    new_time = time.perf_counter()
    global_time_list.append(new_time - global_last_time)
    global_last_time = time.perf_counter()
    # print("step: %.2f"%(global_time_list[-1]*1000))


def last_time():
    """Returns last interval records in millis."""
    global global_last_time, global_time_list
    if global_time_list:
        return 1000 * global_time_list[-1]
    else:
        return 0

def verbose_func(s):
    print(s)


final_loss = None
times = []


def lbfgs(opfunc, x, config, state::State):
    # early exit, when no iteration shall be made
    if config.maxIter == 0:
        return

    global final_loss, times

    maxIter = config.maxIter
    maxEval = config.maxEval or maxIter * 1.25
    tolFun = config.tolFun or 1e-5
    tolX = config.tolX or 1e-19
    nCorrection = config.nCorrection or 100
    lineSearch = config.lineSearch
    lineSearchOpts = config.lineSearchOptions
    learningRate = config.learningRate or 1

    # evaluate initial f(x) and df/dx
    f, g = opfunc(x)
    
    f_hist = [f]
    currentFuncEval = 1
    state.funcEval = state.funcEval + 1
    p = g.shape[0]

    # check optimality of initial point
    tmp1 = tf.abs(g)
    if tf.reduce_sum(tmp1) <= tolFun:
        logging.info("optimality condition below tolFun")
        return x, f_hist

    # variables cached in state (for tracing)
    d = state.d
    t = state.t
    old_dirs = state.old_dirs
    old_stps = state.old_stps
    Hdiag = state.Hdiag
    g_old = state.g_old
    f_old = state.f_old

    # optimize for a max of maxIter iterations
    nIter = 0
    times = []
    while nIter < maxIter:
        start_time = time.time()

        # keep track of nb of iterations
        nIter = nIter + 1
        state.nIter = state.nIter + 1

        ############################################################
        ## compute gradient descent direction
        ############################################################
        if state.nIter == 1:
            d = -g
            old_dirs = []
            old_stps = []
            Hdiag = 1
        else:
            # do lbfgs update (update memory)
            y = g - g_old

            s = d * t
            ys = tf.tensordot(y, s)

            if ys > 1e-10:
                # updating memory
                if len(old_dirs) == nCorrection:
                    # shift history by one (limited-memory)
                    del old_dirs[0]
                    del old_stps[0]

                # store new direction/step
                old_dirs.append(s)
                old_stps.append(y)

                # update scale of initial Hessian approximation
                Hdiag = ys / tf.tensordot(y, y)

            # compute the approximate (L-BFGS) inverse Hessian
            # multiplied by the gradient
            k = len(old_dirs)

            # need to be accessed element-by-element, so don't re-type tensor:
            ro = [0] * nCorrection
            for i in range(k):
                ro[i] = 1 / tf.tensordot(old_stps[i], old_dirs[i])

            # iteration in L-BFGS loop collapsed to use just one buffer
            # need to be accessed element-by-element, so don't re-type tensor:
            al = [0] * nCorrection

            q = -g
            for i in range(k - 1, -1, -1):
                al[i] = tf.tensordot(old_dirs[i], q) * ro[i]
                q = q - al[i] * old_stps[i]

            # multiply by initial Hessian
            r = q * Hdiag
            for i in range(k):
                be_i = tf.tensordot(old_stps[i], r) * ro[i]
                r += (al[i] - be_i) * old_dirs[i]

            d = r
            # final direction is in r/d (same object)

        g_old = g
        f_old = f

        ############################################################
        ## compute step length
        ############################################################
        # directional derivative
        gtd = tf.tensordot(g, d)

        # check that progress can be made along that direction
        if gtd > -tolX:
            logging.warning("Can not make progress along direction.")
            break

        # reset initial guess for step size
        if state.nIter == 1:
            tmp1 = tf.abs(g)
            t = min(1, 1 / tf.reduce_sum(tmp1))
        else:
            t = learningRate

        # optional line search: user function
        lsFuncEval = 0
        if lineSearch and isinstance(lineSearch) == types.FunctionType:
            # perform line search, using user function
            f, g, x, t, lsFuncEval = lineSearch(opfunc, x, t, d, f, g, gtd, lineSearchOpts)
            f_hist.append(f)
        else:
            # no line search, simply move with fixed-step
            x += t * d

            if nIter != maxIter:
                # re-evaluate function only if not in last iteration
                # the reason we do this: in a stochastic setting,
                # no use to re-evaluate that function here
                f, g = opfunc(x)
                lsFuncEval = 1
                f_hist.append(f)

        # update func eval
        currentFuncEval = currentFuncEval + lsFuncEval
        state.funcEval = state.funcEval + lsFuncEval

        ############################################################
        ## check conditions
        ############################################################
        if nIter == maxIter:
            break

        if currentFuncEval >= maxEval:
            # max nb of function evals
            logging.info('The maximum number of function evaluation is reached.')
            break

        tmp1 = tf.abs(g)
        if tf.reduce_sum(tmp1) <= tolFun:
            # check optimality
            logging.info('Optimality condition below tolFun.')
            break

        tmp1 = tf.abs(d * t)
        if tf.reduce_sum(tmp1) <= tolX:
            # step size below tolX
            logging.info('Step size below tolX')
            break

        if tf.abs(f - f_old) < tolX:
            # function value changing less than tolX
            logging.info('Function value changing less than tolX' + str(tf.abs(f - f_old)))
            break

        if logging.getLevelName() > logging.DEBUG:
            logging.debug(nIter, f.numpy(), True)
            # print("Step %3d loss %6.5f msec %6.3f"%(nIter, f.numpy(), last_time()))
            record_time()
            times.append(last_time())

        if nIter == maxIter - 1:
            final_loss = f.numpy()

    # save state
    state.old_dirs = old_dirs
    state.old_stps = old_stps
    state.Hdiag = Hdiag
    state.t = t
    state.d = d
    state.g_old = g_old 
    state.f_old = f_old

    return x, f_hist, currentFuncEval
