"""FinAjan grafik üretimi (charts) birim testleri."""

import pandas as pd
import plotly.graph_objects as go
import pytest

from data.generate_data import generate
from src.charts import (
    category_bar,
    category_trend,
    income_vs_expense,
    monthly_trend,
)


@pytest.fixture(scope="module")
def sample_df():
    df, _ = generate(seed=42)
    return df


def test_category_bar_with_data(sample_df):
    """category_bar fonksiyonunun geçerli bir Plotly figürü döndürdüğünü test eder."""
    fig = category_bar(sample_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].orientation == "h"
    assert "Kategorilere Göre Harcama" in fig.layout.title.text


def test_category_bar_empty():
    """category_bar fonksiyonunun boş veri ile hatasız çalıştığını test eder."""
    fig = category_bar(pd.DataFrame())
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0


def test_monthly_trend_with_data(sample_df):
    """monthly_trend fonksiyonunun gelir, gider ve net bakiye izleri içerdiğini test eder."""
    fig = monthly_trend(sample_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3
    trace_names = [t.name for t in fig.data]
    assert "Gelir" in trace_names
    assert "Gider" in trace_names
    assert "Net Bakiye" in trace_names


def test_monthly_trend_empty():
    """monthly_trend fonksiyonunun boş veri ile hatasız çalıştığını test eder."""
    fig = monthly_trend(pd.DataFrame())
    assert isinstance(fig, go.Figure)


def test_income_vs_expense_with_data(sample_df):
    """income_vs_expense fonksiyonunun 2 bar izi ürettiğini test eder."""
    fig = income_vs_expense(sample_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_income_vs_expense_empty():
    """income_vs_expense fonksiyonunun boş veri ile hatasız çalıştığını test eder."""
    fig = income_vs_expense(pd.DataFrame())
    assert isinstance(fig, go.Figure)


def test_category_trend_with_data(sample_df):
    """category_trend fonksiyonunun seçilen kategori için trend figürü ürettiğini test eder."""
    fig = category_trend(sample_df, "Market")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert "Market" in fig.layout.title.text


def test_category_trend_empty():
    """category_trend fonksiyonunun boş veri veya boş kategoriyle hatasız çalıştığını test eder."""
    fig = category_trend(pd.DataFrame(), "Market")
    assert isinstance(fig, go.Figure)
    fig2 = category_trend(pd.DataFrame(), "")
    assert isinstance(fig2, go.Figure)
