"""FinAjan finansal veri görselleştirme modülü (Plotly figürleri).

Saf fonksiyonlar: Streamlit veya dosya sistemi bağımlılığı içermez.
Tüm fonksiyonlar plotly.graph_objects.Figure nesnesi döndürür.
Boş veya eksik veri durumunda hata vermeden boş figür üretir.
"""

from typing import Union
import pandas as pd
import plotly.graph_objects as go

from src import tools
from src.ui_helpers import month_label


def _empty_figure(title: str, message: str = "Görüntülenecek veri bulunamadı.") -> go.Figure:
    """Veri olmadığında temiz bir yer tutucu figür döndürür."""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_white",
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 14, "color": "#6B7280"},
            }
        ],
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def category_bar(df: pd.DataFrame) -> go.Figure:
    """Kategorilere göre harcama dağılımını gösteren yatay çubuk grafik.

    Args:
        df: İşlem verilerini içeren DataFrame.

    Returns:
        go.Figure: Harcama tutarına göre sıralı yatay bar figürü.
    """
    if df is None or df.empty:
        return _empty_figure("Kategorilere Göre Harcama Dağılımı")

    cat_df = tools.group_by_category(df, expenses_only=True)
    if cat_df.empty:
        return _empty_figure("Kategorilere Göre Harcama Dağılımı")

    # En çok harcanan en üstte görünsün diye sırala
    sorted_df = cat_df.sort_values(by="toplam", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=sorted_df["toplam"],
            y=sorted_df["kategori"],
            orientation="h",
            marker=dict(color="#3B82F6"),
            text=[f"{v:,.2f} TL".replace(",", "X").replace(".", ",").replace("X", ".") for v in sorted_df["toplam"]],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Tutar: %{x:,.2f} TL<extra></extra>",
        )
    )

    fig.update_layout(
        title="Kategorilere Göre Harcama Dağılımı",
        xaxis_title="Toplam Harcama (TL)",
        yaxis_title="Kategori",
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=40),
        height=450,
    )
    return fig


def monthly_trend(df: pd.DataFrame) -> go.Figure:
    """Aylık bazda gelir, gider ve net bakiye trendini gösteren karma grafik.

    Args:
        df: İşlem verilerini içeren DataFrame.

    Returns:
        go.Figure: Gelir, gider ve net bakiye figürü.
    """
    if df is None or df.empty:
        return _empty_figure("Aylık Finansal Trend")

    m_df = tools.monthly_summary(df)
    if m_df.empty:
        return _empty_figure("Aylık Finansal Trend")

    labels = [month_label(a) for a in m_df["ay"]]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=labels,
            y=m_df["gelir"],
            name="Gelir",
            marker_color="#10B981",
            hovertemplate="Gelir: %{y:,.2f} TL<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            x=labels,
            y=m_df["gider"],
            name="Gider",
            marker_color="#EF4444",
            hovertemplate="Gider: %{y:,.2f} TL<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=labels,
            y=m_df["net"],
            name="Net Bakiye",
            mode="lines+markers",
            line=dict(color="#2563EB", width=3),
            marker=dict(size=7),
            hovertemplate="Net: %{y:,.2f} TL<extra></extra>",
        )
    )

    fig.update_layout(
        title="Aylık Gelir, Gider ve Net Bakiye Akışı",
        xaxis_title="Ay",
        yaxis_title="Tutar (TL)",
        barmode="group",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=40),
        height=420,
    )
    return fig


def income_vs_expense(df: pd.DataFrame) -> go.Figure:
    """Aylık gelir ve giderin yan yana karşılaştırmalı çubuk grafiği.

    Args:
        df: İşlem verilerini içeren DataFrame.

    Returns:
        go.Figure: Gruplanmış çubuk grafik figürü.
    """
    if df is None or df.empty:
        return _empty_figure("Aylık Gelir ve Gider Karşılaştırması")

    m_df = tools.monthly_summary(df)
    if m_df.empty:
        return _empty_figure("Aylık Gelir ve Gider Karşılaştırması")

    labels = [month_label(a) for a in m_df["ay"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=m_df["gelir"],
            name="Toplam Gelir",
            marker_color="#059669",
            hovertemplate="%{x}<br>Gelir: %{y:,.2f} TL<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=m_df["gider"],
            name="Toplam Gider",
            marker_color="#DC2626",
            hovertemplate="%{x}<br>Gider: %{y:,.2f} TL<extra></extra>",
        )
    )

    fig.update_layout(
        title="Aylık Gelir - Gider Karşılaştırması",
        xaxis_title="Ay",
        yaxis_title="Tutar (TL)",
        barmode="group",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=40),
        height=400,
    )
    return fig


def category_trend(df: pd.DataFrame, category: str) -> go.Figure:
    """Belirli bir harcama kategorisinin aylık trendini gösteren grafik.

    Args:
        df: İşlem verilerini içeren DataFrame.
        category: İncelenecek harcama kategorisi adı.

    Returns:
        go.Figure: Aylık harcama grafiği.
    """
    if df is None or df.empty or not category:
        return _empty_figure(f"'{category or 'Seçilen'}' Aylık Trendi")

    trend_df = tools.category_monthly_totals(df, category)
    if trend_df.empty:
        return _empty_figure(f"'{category}' Aylık Trendi")

    labels = [month_label(a) for a in trend_df["ay"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=trend_df["toplam"],
            name=category,
            marker_color="#6366F1",
            text=[f"{v:,.2f} TL".replace(",", "X").replace(".", ",").replace("X", ".") for v in trend_df["toplam"]],
            textposition="auto",
            hovertemplate="%{x}<br>Harcama: %{y:,.2f} TL<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"'{category}' Kategorisi Aylık Harcama Trendi",
        xaxis_title="Ay",
        yaxis_title="Harcama Tutarı (TL)",
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=40),
        height=380,
    )
    return fig
