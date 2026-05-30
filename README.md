# Revenue Optimization Simulator

This is an interactive web application built with Streamlit and Pyomo that solves a revenue maximization problem using mathematical optimization (MINLP). It provides intuitive what-if analysis, shadow price evaluation, and sensitivity analysis through dynamic Plotly charts.

**[Try the Live App on Streamlit Community Cloud](https://share.streamlit.io/)** *([Link](https://revenue-optimization-simulator-jgyaappgjvviyf3gtyblkhi.streamlit.app/))*

---

## 🌟 Key Features

- **Optimal Pricing & Demand Calculation**: Instantly computes the price and production volume that maximizes expected profit while respecting capacity limits.
- **Shadow Price Analysis**: Evaluates the hidden value of capacity, minimum price floors, and maximum price ceilings.
- **Sensitivity Analysis (Tornado Chart)**: Visualizes how changes in variable costs, fixed costs, and maximum price limits impact overall profit.
- **Dynamic Demand Curve Visualization**: Interactive Plotly charts mapping out revenue, cost, profit, and the demand curve.
- **Bilingual Support (English / 日本語)**: Toggle between English and Japanese UI dynamically from the sidebar.

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Optimization**: Pyomo
- **Solvers**: SCIP (Primary, via `pyscipopt`), IPOPT (Fallback)
- **Data & Visualization**: Pandas, NumPy, Plotly

## 🚀 How to Run Locally

### 1. Install Dependencies

It is recommended to use a virtual environment. Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App

Execute the following command in your terminal:

```bash
streamlit run app.py
```

The application will open automatically in your default web browser at `http://localhost:8501`.

---

# 収益最大化シミュレーター (Revenue Optimization Simulator)

本アプリケーションは、StreamlitとPyomoを用いて構築された、数理最適化（MINLP）による収益最大化問題を解くインタラクティブなWebアプリケーションです。直感的なWhat-If分析、シャドープライス評価、および動的なPlotlyチャートによる感度分析を提供します。

**[Streamlit Community Cloud でアプリを試す](https://share.streamlit.io/)** *(デプロイ後にリンクを更新してください)*

---

## 🌟 主な機能

- **最適価格と需要の計算**: 生産能力の限界を考慮しつつ、期待利益を最大化する最適な価格と生産量を瞬時に計算します。
- **シャドープライス分析**: 生産能力、最低価格、最高価格の隠れた価値（ボトルネック解消時の利益増加額）を評価します。
- **感度分析（トルネードチャート）**: 変動費、固定費、最高価格上限の変化が全体利益に与える影響を可視化します。
- **動的な需要曲線と利益構造の可視化**: 売上、コスト、利益、および需要曲線をPlotlyチャートでインタラクティブに表示します。
- **バイリンガル対応（英語 / 日本語）**: サイドバーから英語と日本語のUIを動的に切り替えることができます。

## 🛠️ 使用技術

- **フロントエンド**: Streamlit
- **数理最適化モデリング**: Pyomo
- **ソルバー**: SCIP (優先, `pyscipopt` 経由), IPOPT (フォールバック)
- **データ・可視化**: Pandas, NumPy, Plotly

## 🚀 ローカルでの実行方法

### 1. 依存パッケージのインストール

仮想環境の使用を推奨します。以下のコマンドで必要なPythonパッケージをインストールしてください。

```bash
pip install -r requirements.txt
```

### 2. Streamlit アプリの起動

ターミナルで以下のコマンドを実行します。

```bash
streamlit run app.py
```

デフォルトのWebブラウザが自動的に開き、`http://localhost:8501` でアプリケーションが表示されます。
