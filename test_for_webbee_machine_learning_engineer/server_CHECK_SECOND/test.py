import logging

import pandas as pd
import pytest
import requests


@pytest.fixture
def server_url():
    yield 'https://sirpoopy.pythonanywhere.com/web_bee_predict_sale_price'


def test_right_simple_json_file(server_url):
    test_df = pd.read_csv('../test_task_CHECK_FIRST/data/test.csv')

    response = requests.post(server_url, files={"file": test_df.to_json()})

    assert response.status_code == 200


def test_respond_amount_entities(server_url):
    test_df = pd.read_csv('../test_task_CHECK_FIRST/data/test.csv')

    response = requests.post(server_url, files={"file": test_df.to_json()})

    assert len(test_df) == len(eval(response.text))


# We can check a lot of thing here, but main idea I realized
