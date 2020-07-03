import pytest


def test_regressor():
    x = load_tick_data("path_to_tick_data")
    buy_price = buy_regressor(x)
    sell_price = sell_regressor(x)

