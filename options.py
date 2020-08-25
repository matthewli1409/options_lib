import math

import numpy as np

from scipy.stats import norm


def black_price(spot_px, strike_px, rf_rate, ttm, vol, opt_type):
    """Black Scholes Option Price

    Args:
        spot_px (float): spot price
        strike_px (float): strike price
        rf_rate (float): risk-free rate
        ttm (float): time to maturity in years
        vol (float): volatility
        opt_type (str): 'call' or 'put'

    Returns:
        (float): Black Scholes Option Price
    """
    if ttm <= 0:
        return 0

    d1 = (np.log(spot_px / strike_px) + ttm *
          (0.5 * math.pow(vol, 2))) / (vol * math.sqrt(ttm))
    d2 = d1 - vol * math.sqrt(ttm)

    if opt_type == 'call':
        return math.exp(-rf_rate * ttm) * (spot_px * norm.cdf(d1) - strike_px * norm.cdf(d2))
    elif opt_type == 'put':
        return math.exp(-rf_rate * ttm) * (strike_px * norm.cdf(-d2) - spot_px * norm.cdf(-d1))


def black_delta(spot_px, strike_px, rf_rate, ttm, vol, opt_type):
    """Black Scholes Delta Model

    Args:
        spot_px (float): spot price
        strike_px (float): strike price
        rf_rate (float): risk-free rate
        ttm (float): time to maturity in years
        vol (float): volatility
        opt_type (str): 'call' or 'put'

    Returns:
        (float): Black Scholes Delta
    """
    if ttm <= 0:
        return 0

    d1 = (np.log(spot_px / strike_px) + ttm *
          (0.5 * math.pow(vol, 2))) / (vol * math.sqrt(ttm))

    if opt_type == 'call':
        return math.exp(-rf_rate * ttm) * norm.cdf(d1)
    elif opt_type == 'put':
        return math.exp(-rf_rate * ttm) * (norm.cdf(d1) - 1)


def black_gamma(spot_px, strike_px, rf_rate, ttm, vol):
    """Black Scholes Gamma Model

    Args:
        spot_px (float): spot price
        strike_px (float): strike price
        rf_rate (float): risk-free rate
        ttm (float): time to maturity in years
        vol (float): volatility

    Returns:
        (float): Black Scholes Gamma
    """
    if ttm <= 0:
        return 0

    d1 = (np.log(spot_px / strike_px) + ttm *
          (0.5 * math.pow(vol, 2))) / (vol * math.sqrt(ttm))
    return math.exp(-rf_rate * ttm) * (norm.pdf(d1) / (spot_px * vol * math.sqrt(ttm)))


def black_vega(spot_px, strike_px, rf_rate, ttm, vol):
    """Black Scholes Vega Model

    Args:
        spot_px (float): spot price
        strike_px (float): strike price
        rf_rate (float): risk-free rate
        ttm (float): time to maturity in years
        vol (float): volatility

    Returns:
        (float): Black Scholes Vega
    """
    if ttm <= 0:
        return 0

    d1 = (np.log(spot_px / strike_px) + ttm *
          (0.5 * math.pow(vol, 2))) / (vol * math.sqrt(ttm))
    return (spot_px * math.exp(-rf_rate * ttm) * norm.pdf(d1) * math.sqrt(ttm)) / 100


def black_theta(spot_px, strike_px, rf_rate, ttm, vol, opt_type):
    """Black Scholes Theta Model

    Args:
        spot_px (float): spot price
        strike_px (float): strike price
        rf_rate (float): risk-free rate
        ttm (float): time to maturity in years
        vol (float): volatility
        opt_type (str): 'call' or 'put'

    Returns:
        (float): Black Scholes Theta
    """
    if ttm <= 0:
        return 0

    d1 = (np.log(spot_px / strike_px) + ttm *
          (0.5 * math.pow(vol, 2))) / (vol * math.sqrt(ttm))
    d2 = d1 - vol * math.sqrt(ttm)

    if opt_type == 'call':
        return abs(-spot_px * math.exp(-rf_rate * ttm) * norm.pdf(d1) * vol / (2 * math.sqrt(ttm)) + rf_rate * spot_px * math.exp(-rf_rate * ttm) * norm.cdf(d1) - rf_rate * strike_px * math.exp(-rf_rate * ttm) * norm.cdf(d2))/365
    elif opt_type == 'put':
        return abs(-spot_px * math.exp(-rf_rate * ttm) * norm.pdf(d1) * vol / (2 * math.sqrt(ttm)) - rf_rate * spot_px * math.exp(-rf_rate * ttm) * norm.cdf(d1) + rf_rate * strike_px * math.exp(-rf_rate * ttm) * norm.cdf(d2))/365


def implied_vol(rf_rate, spot_px, strike_px, opt_price, ttm, opt_type):
    """Get implied volatility by interpolating till price is near the black price

    Notes:
        - This is a very inefficient way of doing it

    Args:
        spot_px (float): spot price
        strike_px (float): strike price
        rf_rate (float): risk-free rate
        ttm (float): time to maturity in years
        vol (float): volatility
        opt_type (str): 'call' or 'put'

    Returns:
        (float): Black Scholes Theta

    """
    vol = 1
    while (abs(black_price(spot_px, strike_px, rf_rate, ttm, vol, opt_type) - opt_price) > 1.0):
        if black_price(spot_px, strike_px, rf_rate, ttm, vol, opt_type) > opt_price:
            vol -= 0.01
        else:
            vol += 0.01
    return vol


if __name__ == '__main__':

    # k = 95
    # s = 100
    # r = 0
    # t = 30/365
    # v = 0.25

    # black_price_c = black_price(k, s, r, t, v, 'call')
    # black_price_p = black_price(k, s, r, t, v, 'put')
    # black_delta_c = black_delta(k, s, r, t, v, 'call')
    # black_delta_p = black_delta(k, s, r, t, v, 'put')
    # black_gamma = black_gamma(k, s, r, t, v)
    # black_vega = black_vega(k, s, r, t, v)
    # black_theta_c = black_theta(k, s, r, t, v, 'call')
    # black_theta_p = black_theta(k, s, r, t, v, 'put')

    # print(f'call price: {black_price_c}')
    # print(f'put price: {black_price_p}')
    # print(f'call delta: {black_delta_c}')
    # print(f'put delta: {black_delta_p}')
    # print(f'gamma: {black_gamma}')
    # print(f'vega {black_vega}')
    # print(f'call theta: {black_theta_c}')
    # print(f'put theta: {black_theta_p}')

    t = 3/365
    print(black_price(11600, 10000, 0, 3/365, 0.8, 'call'))
    iv = implied_vol(0, 11600, 10000, 1767, t, 'call')
    print(iv)
