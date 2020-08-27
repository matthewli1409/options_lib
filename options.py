import math

import numpy as np

from scipy.stats import norm


def black_price(spot_px, strike_px, rf_rate, div_yield, ttm, vol, opt_type):
    """Black Scholes Option Price

    Args:
        spot_px (float): spot price
        strike_px (float): strike price
        rf_rate (float): risk-free rate
        div_yield (float): dividend yield rate
        ttm (float): time to maturity in years
        vol (float): volatility
        opt_type (str): 'call' or 'put'

    Returns:
        (float): Black Scholes Option Price
    """
    if ttm <= 0:
        return 0

    d1 = (np.log(spot_px / strike_px) + ttm *
          (0.5 * math.pow(vol, 2) + rf_rate - div_yield)) / (vol * math.sqrt(ttm))
    d2 = d1 - vol * math.sqrt(ttm)

    if opt_type == 'call':
        return (math.exp(-div_yield * ttm) * spot_px * norm.cdf(d1)) - (math.exp(-rf_rate * ttm) * strike_px * norm.cdf(d2))
    elif opt_type == 'put':
        return (math.exp(-rf_rate * ttm) * strike_px * norm.cdf(-d2)) - (math.exp(-div_yield * ttm) * spot_px * norm.cdf(-d1))


def black_delta(spot_px, strike_px, rf_rate, div_yield, ttm, vol, opt_type):
    """Black Scholes Delta Model

    Args:
        spot_px (float): spot price
        strike_px (float): strike price
        rf_rate (float): risk-free rate
        div_yield (float): dividend yield rate
        ttm (float): time to maturity in years
        vol (float): volatility
        opt_type (str): 'call' or 'put'

    Returns:
        (float): Black Scholes Delta
    """
    if ttm <= 0:
        return 0

    d1 = (np.log(spot_px / strike_px) + ttm *
          (0.5 * math.pow(vol, 2) + rf_rate - div_yield)) / (vol * math.sqrt(ttm))

    if opt_type == 'call':
        return math.exp(-div_yield * ttm) * norm.cdf(d1)
    elif opt_type == 'put':
        return -math.exp(-rf_rate * ttm) * (norm.cdf(-d1))


def black_gamma(spot_px, strike_px, rf_rate, div_yield, ttm, vol):
    """Black Scholes Gamma Model

    Args:
        spot_px (float): spot price
        strike_px (float): strike price
        rf_rate (float): risk-free rate
        div_yield (float): dividend yield rate
        ttm (float): time to maturity in years
        vol (float): volatility

    Returns:
        (float): Black Scholes Gamma
    """
    if ttm <= 0:
        return 0

    d1 = (np.log(spot_px / strike_px) + ttm *
          (0.5 * math.pow(vol, 2) + rf_rate - div_yield)) / (vol * math.sqrt(ttm))
    return math.exp(-div_yield * ttm) * (norm.pdf(d1) / (spot_px * vol * math.sqrt(ttm)))


def black_vega(spot_px, strike_px, rf_rate, div_yield, ttm, vol):
    """Black Scholes Vega Model

    Args:
        spot_px (float): spot price
        strike_px (float): strike price
        rf_rate (float): risk-free rate
        div_yield (float): dividend yield rate
        ttm (float): time to maturity in years
        vol (float): volatility

    Returns:
        (float): Black Scholes Vega
    """
    if ttm <= 0:
        return 0

    d1 = (np.log(spot_px / strike_px) + ttm *
          (0.5 * math.pow(vol, 2) + rf_rate - div_yield)) / (vol * math.sqrt(ttm))
    return (spot_px * math.exp(-div_yield * ttm) * norm.pdf(d1) * math.sqrt(ttm)) / 100


def black_theta(spot_px, strike_px, rf_rate, div_yield, ttm, vol, opt_type):
    """Black Scholes Theta Model

    Args:
        spot_px (float): spot price
        strike_px (float): strike price
        rf_rate (float): risk-free rate
        div_yield (float): dividend yield rate
        ttm (float): time to maturity in years
        vol (float): volatility
        opt_type (str): 'call' or 'put'

    Returns:
        (float): Black Scholes Theta
    """
    if ttm <= 0:
        return 0

    d1 = (np.log(spot_px / strike_px) + ttm *
          (0.5 * math.pow(vol, 2) + rf_rate - div_yield)) / (vol * math.sqrt(ttm))
    d2 = d1 - vol * math.sqrt(ttm)

    if opt_type == 'call':
        return (-spot_px * math.exp(-div_yield * ttm) * norm.pdf(d1) * vol / (2 * math.sqrt(ttm)) + (div_yield * spot_px * math.exp(-div_yield * ttm) * norm.cdf(d1)) - (rf_rate * strike_px * math.exp(-rf_rate * ttm) * norm.cdf(d2)))/365
    elif opt_type == 'put':
        return (-spot_px * math.exp(-div_yield * ttm) * norm.pdf(-d1) * vol / (2 * math.sqrt(ttm)) - (div_yield * spot_px * math.exp(-div_yield * ttm) * norm.cdf(-d1)) + (rf_rate * strike_px * math.exp(-rf_rate * ttm) * norm.cdf(-d2)))/365


def implied_vol(spot_px, strike_px, rf_rate, div_yield, ttm, opt_type, opt_price):
    """Get implied volatility by interpolating till price is near the black price

    Notes:
        - This is a very inefficient way of doing it

    Args:
        spot_px (float): spot price
        strike_px (float): strike price
        rf_rate (float): risk-free rate
        div_yield (float): dividend yield rate
        ttm (float): time to maturity in years
        opt_type (str): 'call' or 'put
        opt_price (float): the price of the option

    Returns:
        (float): Black Scholes Theta
    """
    vol = 1
    k = 10000000000
    while abs(black_price(spot_px, strike_px, rf_rate, div_yield, ttm, vol, opt_type) - opt_price) > 1.0:
        if black_price(spot_px, strike_px, rf_rate, div_yield, ttm, vol, opt_type) > opt_price:
            proceed = False
            while not proceed:
                new_potential_black_price = black_price(
                    spot_px, strike_px, rf_rate, div_yield, ttm, vol - k, opt_type)
                if new_potential_black_price < opt_price:
                    # too big
                    k = k / 10
                else:
                    proceed = True
            vol -= k
        else:
            proceed = False
            while not proceed:
                new_potential_black_price = black_price(
                    spot_px, strike_px, rf_rate, div_yield, ttm, vol - k, opt_type)
                if new_potential_black_price > opt_price:
                    # too big
                    k = k / 10
                else:
                    proceed = True
            vol += k
    return vol


if __name__ == '__main__':

    strike = 95
    underlying = 100
    div_yield = 0.0
    rf_rate = 0.0
    ttm = 30/365
    vol = 0.25

    print(black_price(underlying, strike, rf_rate, div_yield, ttm, vol, 'call'))
    print(black_price(underlying, strike, rf_rate, div_yield, ttm, vol, 'put'))

    print(black_delta(underlying, strike, rf_rate, div_yield, ttm, vol, 'call'))
    print(black_delta(underlying, strike, rf_rate, div_yield, ttm, vol, 'put'))

    print(black_theta(underlying, strike, rf_rate, div_yield, ttm, vol, 'call'))
    print(black_theta(underlying, strike, rf_rate, div_yield, ttm, vol, 'put'))

    print(black_gamma(underlying, strike, rf_rate, div_yield, ttm, vol))
    print(black_vega(underlying, strike, rf_rate, div_yield, ttm, vol))

    strike = 10000
    underlying = 11400
    div_yield = 0
    rf_rate = 0
    ttm = 3/365

    print(implied_vol(underlying, strike, rf_rate, div_yield, ttm, 'call', 1500))
