import numpy as np
import pandas as pd
import xarray as xr
np.seterr(divide='ignore', invalid='ignore')
"""Contains verification metrics"""


def RMSE(pred, obs):
    return float(np.sqrt(np.nanmean((pred - obs)**2)))


def ME(pred, obs):
    return float(np.nanmean(pred - obs))


def NSE(pred, obs):
    """Calculate the Nash-sutcliffe efficiency.

    Denominator: variance of predicting the mean value"""
    difference = pred - obs
    squarediff = np.dot(difference, difference)
    obs_dis_anom = obs - np.nanmean(obs)
    return float(1-squarediff/np.dot(obs_dis_anom, obs_dis_anom))


def NSE_diff(pred, obs):
    """Calculate the Nash-sutcliffe efficiency for models trained on differences.

    Denominator: variance of predicting the mean value of differences.

    This variation of the NSE takes into account that the most appropriate
    baseline forecast is persistence when predicting differences.

    TODO: this function assumes pred.forecast_day.values is scalar, what if not?

    Parameters
    ----------
    pred : xr.DataArray
        the prediction array
    obs : xr.DataArray
        the true values to compare with

    Returns
    -------
    (float, float)
        1) Variance reduction due compared to persistence forecasts
        2) NSE of persistence forecasts (is a variance reduction too)
    """
    inits = pred.init_time.values

    # for subtracting observations, the time coordinate has to be its valid time
    pred = pred.swap_dims({'init_time': 'time'})
    diff = pred - obs

    # now we will iterate over the init_time, so this shall be our index
    diff = diff.swap_dims({'time': 'init_time'})
    pred = pred.swap_dims({'time': 'init_time'})

    err = err_persistence = err_mean_dis = 0
    for init in inits:
        valid_time = pred.sel(init_time=init).time

        # Variance Reduction compared to persistence forecasts
        d = diff.sel(init_time=init)  # is scalar if using one fcstday!
        err += float(xr.dot(d, d))

        persistence = obs.sel(time=init)
        d = persistence - obs.sel(time=valid_time)
        err_persistence += float(xr.dot(d, d))

        # NSE of persistence forecasts
        d = obs.mean() - obs.sel(time=valid_time)
        err_mean_dis += float(xr.dot(d, d))

    return float(1-err/err_persistence), float(1-err_persistence/err_mean_dis)


def RMSE_persistence(pred, obs):
    """Calculate the RMSE for persistence forecasts.

    Parameters
    ----------
    pred : xr.DataArray
        the prediction array, from which the timestamps are taken
        to verify the persistence forecast on
    obs : xr.DataArray
        the true values to compare with

    TODO: now assuming pred.forecast_day.values is scalar
    """
    inits = pred.init_time.values

    err = np.zeros(len(inits))
    for i, init in enumerate(inits):
        valid_time = pred.sel(init_time=init).time

        persistence = obs.sel(time=init)
        d = persistence - obs.sel(time=valid_time)
        err[i] = float(xr.dot(d, d))

    return np.sqrt(np.mean(err))


def verify(prediction, truth):
    """Evaluate multiday forecasts.

    As we are dealing with multiday forecasts,
    we will evaluate each forecast day separately.

    Parameters
    ----------
    prediction : xr.DataArray
        2-dimensional discharge forecast (init_time, forecast_day)

    truth : xr.DataArray
        1-dimensional discharge (time)
    """
    fcst_days = prediction.forecast_day.values

    # allocate to save scores
    N_scores = 6
    scores = np.full((len(fcst_days), N_scores+1,), np.nan)

    # separate scores for each forecast day
    for i, day in enumerate(fcst_days):
        print('forecast-day:', day)

        # get prediction values for these days (multiple inits)
        pred_fxd = prediction.sel(forecast_day=day)

        # get observation values for these days
        valid_time = pred_fxd.init_time + np.timedelta64(day, 'D')
        truth_fxd = truth.sel(time=valid_time)

        var_red_persistence, nse_persistence = NSE_diff(pred_fxd, truth)

        scores[i, :] = np.array([day,
                                ME(pred_fxd, truth_fxd),
                                RMSE(pred_fxd, truth_fxd),
                                RMSE_persistence(pred_fxd, truth),
                                NSE(pred_fxd, truth_fxd),
                                nse_persistence,
                                var_red_persistence])

    df = pd.DataFrame(index=scores[:, 0], data=scores[:, 1:],
                      columns=['ME', 'RMSE', 'RMSE_persistence', 'NSE',
                               'NSE_persistence', 'VR_persistence'])
    df.index.name = 'forecast_day'
    return df
