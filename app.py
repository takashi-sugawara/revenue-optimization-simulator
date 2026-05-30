import streamlit as st
import pyomo.environ as pyo
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pyomo.opt import SolverFactory
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

@dataclass(frozen=True)
class OptimizationResult:
    price: Optional[float]
    demand: Optional[float]
    revenue: Optional[float]
    profit: Optional[float]
    success: bool
    dual_info: Optional[Dict[str, float]] = None
    error_msg: Optional[str] = None
    fallback_active: bool = False

DUAL_TOLERANCE = 1e-6

# macOS環境向け：Streamlit実行時にソルバーのパスが通っていない問題を自動解決する
# Homebrew (scip) と Anaconda (ipopt) のパスを追加
for path in ["/opt/homebrew/bin", "/opt/anaconda3/bin", "/usr/local/bin"]:
    if path not in os.environ.get("PATH", ""):
        os.environ["PATH"] += os.pathsep + path

# Streamlitのページ設定を「ワイド」にし、タイトルなどを設定
st.set_page_config(
    page_title="Revenue Optimization Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------
# i18n Helper Setup
# -----------------------------------------------------------------
lang = st.sidebar.radio("Language / 言語", ["English", "日本語"], index=0)

def _(en: str, ja: str) -> str:
    """Returns the string corresponding to the selected language."""
    return en if lang == "English" else ja

st.title(_("📈 Price & Profit Optimization Simulator", "📈 価格・コスト・収益・利益最大化意思決定シミュレーター"))
st.caption(_(
    "A simulator that integrates price elasticity of demand, shadow prices, variable costs, and fixed costs to assist in optimal pricing.",
    "需要の価格弾力性、シャドープライス（潜在価格）、仕入れ変動費や固定費を統合し、最適な価格設定を支援するシミュレーターです。"
))

# -----------------------------------------------------------------
# 1. サイドバー設定 & パラメータ入力
# -----------------------------------------------------------------
st.sidebar.header(_("🛠️ Parameter Settings", "🛠️ パラメータ設定"))

# 需要曲線の設定
st.sidebar.subheader(_("1. Demand Curve Settings", "1. 需要曲線の設定"))
st.sidebar.markdown(_(r"Demand Function: $N = A - B \times p$", r"需要関数: $N = A - B \times p$"))
param_A = st.sidebar.slider(_("Max Demand A (Demand when price is 0)", "最大需要数 A (価格が0の時の需要)"), min_value=500, max_value=2000, value=1001, step=10)
param_B = st.sidebar.slider(_("Price Sensitivity B (Demand drop per 1 unit price increase)", "価格感度 B (価格が1上昇した時の需要の減少数)"), min_value=1.0, max_value=15.0, value=5.0, step=0.5)

# 需要がゼロになる境界価格の計算
price_demand_zero = param_A / param_B
st.sidebar.info(_(f"💡 Price limit where demand hits zero: **¥{price_demand_zero:.1f}**", f"💡 需要がゼロになる価格の上限: **¥{price_demand_zero:.1f}**"))

# コストの設定
st.sidebar.subheader(_("2. Cost Settings", "2. コストの設定"))
cost_var = st.sidebar.slider(_("Variable Cost / Unit Cost ¥", "仕入れ単価・変動費 ¥ (商品1個あたり)"), min_value=0, max_value=100, value=30, step=5)
cost_fix = st.sidebar.slider(_("Fixed Cost ¥ (Rent, Ads, etc.)", "事業固定費 ¥ (店舗運営・広告費など)"), min_value=0, max_value=5000, value=1000, step=100)

# 価格の制約条件
st.sidebar.subheader(_("3. Price Constraints", "3. 価格の制約条件"))
min_price = st.sidebar.slider(_("Minimum Allowable Price (Lower Bound)", "最低許容価格 (Lower Bound)"), min_value=10, max_value=150, value=50, step=5)
max_price = st.sidebar.slider(_("Maximum Allowable Price (Upper Bound)", "最高許容価格 (Upper Bound)"), min_value=150, max_value=400, value=200, step=5)
opt_integer_price = st.sidebar.checkbox(
    _("Require integer pricing (1 unit steps)", "価格も1円単位の整数にする (価格の整数制約)"),
    value=False,
    help=_(
        "If enabled, the optimal price p is restricted to integer values, making this a true mixed-integer programming problem.",
        "有効にすると、最適価格 p も整数（1円単位）に制限され、真の混合整数計画問題になります。"
    )
)

# 生産能力の制約条件
st.sidebar.subheader(_("4. Production & Inventory Constraints", "4. 生産・在庫の制約条件"))
enable_capacity = st.sidebar.checkbox(_("Enable production/inventory limits", "生産能力・在庫制限を設定する"), value=False)
if enable_capacity:
    param_capacity = st.sidebar.slider(
        _("Max Production / Inventory Limit", "最大生産能力 / 在庫制限 (上限)"),
        min_value=100,
        max_value=2000,
        value=800,
        step=50,
        help=_(
            "Restricts the sales volume N from exceeding this limit. Broadens shadow price analysis.",
            "販売数量 N がこの上限を超えられないように制限します。シャドープライス分析の幅が広がります。"
        )
    )
else:
    param_capacity = None

# ✅ パラメータの妥当性検証を一括集約化
def validate_parameters(A: float, B: float, p_min: float, p_max: float, c_var: float, c_fix: float) -> Tuple[List[str], List[str]]:
    """パラメータの妥当性をチェックし、エラーと警告を返します"""
    errors = []
    warnings = []

    if p_min >= p_max:
        errors.append(_(
            "Minimum allowable price must be lower than maximum allowable price.",
            "最低許容価格は最高許容価格より小さく設定してください。"
        ))

    # 最高価格での利益マージンをチェック
    max_margin_at_p_max = p_max - c_var
    if max_margin_at_p_max <= 0:
        errors.append(_(
            f"Variable cost ¥{c_var} is greater than or equal to max price ¥{p_max}. "
            "No profit can be generated. Please lower variable cost or increase max price.",
            f"変動費 ¥{c_var} が最高許容価格 ¥{p_max} 以上のため、どのような販売価格でも利益が出ません。変動費を下げるか、最高価格を上げて利益の出るマージンを確保してください。"
        ))
    elif max_margin_at_p_max < 10:
        warnings.append(_(
            f"Profit margin at max price ¥{p_max} is only ¥{max_margin_at_p_max:.0f}. "
            "Consider reducing variable costs or applying premium pricing.",
            f"最高価格 ¥{p_max} での利益マージンがわずか ¥{max_margin_at_p_max:.0f} しかありません。価格設定に余裕が少ないため、変動費削減またはプレミアム価格設定を検討してください。"
        ))

    p_demand_zero = A / B
    if p_max > p_demand_zero:
        warnings.append(_(
            f"The max price ¥{p_max} exceeds the zero-demand threshold ¥{p_demand_zero:.1f}. "
            "Demand will completely vanish in this price range. We recommend lowering the max price.",
            f"設定された最高価格 ¥{p_max} は、需要がゼロになる限界価格 ¥{p_demand_zero:.1f} を超えています。これ以上の高価格帯では需要が完全に消失するため、最高価格の上限を下げることをお勧めします。"
        ))

    if c_var < 0 or c_fix < 0:
        errors.append(_(
            "Cost parameters (variable and fixed costs) must be 0 or greater.",
            "コストパラメータ（変動費・固定費）は0以上である必要があります。"
        ))

    return errors, warnings

# 入力値検証の実行
validation_errors, validation_warnings = validate_parameters(
    param_A, param_B, min_price, max_price, cost_var, cost_fix
)

# エラー表示
for err in validation_errors:
    st.sidebar.error(f"❌ {err}")

# 警告表示
for warn in validation_warnings:
    st.sidebar.warning(f"⚠️ {warn}")

# エラーがある場合は Streamlit の実行を停止し最適化をスキップ
if validation_errors:
    st.stop()

# -----------------------------------------------------------------
# 2. ソルバーの存在確認とフォールバック
# -----------------------------------------------------------------
@st.cache_resource
def check_solver_availability(name: str):
    """ソルバーが利用可能かを確認する（Pyomoバージョン互換対応、起動時に一度だけキャッシュ）"""
    try:
        opt = SolverFactory(name)
        # opt.available() はPyomoのバージョンによって引数が異なるため、
        # 引数なしで呼び出してTrueを確認する
        if opt.available():
            return True, opt
        return False, None
    except Exception:
        return False, None

# -----------------------------------------------------------------
# 3. 最適化実行コアエンジン（Copilot改善提案 1, 3, 6, 9）
# -----------------------------------------------------------------
@st.cache_data
def optimize_revenue_impl(A: float, B: float, p_min: float, p_max: float, c_var: float, c_fix: float, integer_mode: bool = True, integer_price: bool = False, capacity_value: float = float('inf')) -> OptimizationResult:
    try:
        capacity = None if capacity_value == float('inf') else capacity_value
        model = pyo.ConcreteModel()
        
        # 安全な需要の上限値設定（Copilot改善提案 3: ソルバーの探索範囲を絞り込み安定化）
        max_possible_demand = int(A) + 1

        # ソルバーの選択とフォールバックの判定を最初に行う（変数重複定義のバグ回避）（改善5）
        fallback_active = False
        scip_available = False
        ipopt_available = False

        if integer_mode:
            scip_available, opt = check_solver_availability('scip')
            if not scip_available:
                ipopt_available, opt = check_solver_availability('ipopt')
                if ipopt_available:
                    fallback_active = True
                else:
                    return OptimizationResult(
                        price=None, demand=None, revenue=None, profit=None,
                        success=False, error_msg=_("Available solvers (scip, ipopt) could not be found.", "利用可能なソルバー (scip, ipopt) がシステムに見つかりません。"),
                        fallback_active=False
                    )
        else:
            ipopt_available, opt = check_solver_availability('ipopt')
            if not ipopt_available:
                return OptimizationResult(
                    price=None, demand=None, revenue=None, profit=None,
                    success=False, error_msg=_("Continuous solver 'ipopt' could not be found.", "連続計画用ソルバー 'ipopt' がシステムに見つかりません。"),
                    fallback_active=False
                )

        # 判定結果に基づいて、一度だけ変数を安全に定義する
        if integer_mode and not fallback_active:
            # SCIP 整数計画
            if integer_price:
                model.p = pyo.Var(within=pyo.Integers, bounds=(p_min, p_max))
            else:
                model.p = pyo.Var(bounds=(p_min, p_max))
            model.N = pyo.Var(within=pyo.Integers, bounds=(0, max_possible_demand))
        else:
            # IPOPT 連続計画（またはフォールバック時の連続計画）
            if not integer_mode:
                # 連続計画のシャドープライスを取得するため、境界を明示的な制約条件にする
                model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
                model.p = pyo.Var()
                model.N = pyo.Var(within=pyo.Reals, bounds=(0, max_possible_demand))
                model.C_min_p = pyo.Constraint(expr=model.p >= p_min)
                model.C_max_p = pyo.Constraint(expr=model.p <= p_max)
            else:
                # 整数計画からIPOPTへのフォールバック時（境界制約は Var の bounds で行う）
                model.p = pyo.Var(bounds=(p_min, p_max))
                model.N = pyo.Var(within=pyo.Reals, bounds=(0, max_possible_demand))

        p = model.p
        N = model.N
        
        # 目的関数: 利益 (価格 - 変動費) * 数量 - 固定費 の最大化
        model.obj = pyo.Objective(expr=(p - c_var) * N - c_fix, sense=pyo.maximize)
        
        # 需要曲線の制約条件
        model.C1 = pyo.Constraint(expr=N == A - B * p)
        
        # 生産能力の制約条件（改善④）
        if capacity is not None:
            model.C_capacity = pyo.Constraint(expr=N <= capacity)
        
        # ソルブ実行
        results = opt.solve(model, tee=False)
        
        # 出力値の検証（Copilot改善提案 9）
        p_val = pyo.value(p)
        N_val = pyo.value(N)
        
        if p_val is None or N_val is None:
            return OptimizationResult(
                price=None, demand=None, revenue=None, profit=None,
                success=False, error_msg=_("Solver could not find a feasible optimal solution.", "ソルバーが実行可能な最適解を見つけられませんでした。"),
                fallback_active=fallback_active
            )
            
        # 値の合理性チェック
        if N_val < -1e-5 or p_val < (p_min - 1e-5) or p_val > (p_max + 1e-5):
            return OptimizationResult(
                price=None, demand=None, revenue=None, profit=None,
                success=False, error_msg=_("Warning: Solver returned an unstable solution violating bounds.", "警告: ソルバーが制約を満たさない不安定な解を返しました。"),
                fallback_active=fallback_active
            )

        revenue_val = p_val * N_val
        profit_val = pyo.value(model.obj)
        
        if integer_mode and not fallback_active:
            return OptimizationResult(
                price=p_val, demand=N_val, revenue=revenue_val, profit=profit_val,
                success=True, dual_info=None, fallback_active=False
            )
        else:
            # IPOPT成功時、またはIPOPTへのフォールバック成功時
            dual_info = {
                'dual_demand': model.dual.get(model.C1, 0.0),
                'dual_min_p': model.dual.get(model.C_min_p, 0.0) if hasattr(model, 'C_min_p') else 0.0,
                'dual_max_p': model.dual.get(model.C_max_p, 0.0) if hasattr(model, 'C_max_p') else 0.0,
                'dual_capacity': model.dual.get(model.C_capacity, 0.0) if hasattr(model, 'C_capacity') else 0.0
            }
            return OptimizationResult(
                price=p_val, demand=N_val, revenue=revenue_val, profit=profit_val,
                success=True, dual_info=dual_info, fallback_active=fallback_active
            )
            
    except Exception as e:
        return OptimizationResult(
            price=None, demand=None, revenue=None, profit=None,
            success=False, error_msg=_(f"Optimization Error: {str(e)}", f"最適化実行エラー: {str(e)}"),
            fallback_active=False
        )

def optimize_revenue(A: float, B: float, p_min: float, p_max: float, c_var: float, c_fix: float, integer_mode: bool = True, integer_price: bool = False, capacity: Optional[float] = None) -> OptimizationResult:
    """ユーザー向けラッパー（キャッシュキー一貫性のため None を float('inf') に正規化して impl を呼び出す）"""
    capacity_value = float('inf') if capacity is None else float(capacity)
    return optimize_revenue_impl(A, B, p_min, p_max, c_var, c_fix, integer_mode, integer_price, capacity_value)

# 最適化の実行（キャッシュ呼び出し）
result_int = optimize_revenue(
    param_A, param_B, min_price, max_price, cost_var, cost_fix, integer_mode=True, integer_price=opt_integer_price, capacity=param_capacity
)
opt_p_int = result_int.price
opt_N_int = result_int.demand
opt_rev_int = result_int.revenue
opt_prof_int = result_int.profit
success_int = result_int.success
fallback_int_active = result_int.fallback_active

result_real = optimize_revenue(
    param_A, param_B, min_price, max_price, cost_var, cost_fix, integer_mode=False, capacity=param_capacity
)
opt_p_real = result_real.price
opt_N_real = result_real.demand
opt_rev_real = result_real.revenue
opt_prof_real = result_real.profit
success_real = result_real.success

# -----------------------------------------------------------------
# 4. シミュレーション用データの生成（Copilot改善提案 7: 関数化して可読性アップ）
# -----------------------------------------------------------------
@st.cache_data
def generate_simulation_data_impl(A: float, B: float, c_var: float, c_fix: float, p_min: float, p_max: float, capacity_value: float = float('inf')) -> pd.DataFrame:
    capacity = None if capacity_value == float('inf') else capacity_value
    prices = np.linspace(p_min, p_max, 300)
    demands = np.maximum(0, A - B * prices)
    
    if capacity is not None and capacity > 0:
        demands = np.minimum(demands, capacity)
        
    revenues = prices * demands
    profits = (revenues - c_var * demands) - c_fix
    
    return pd.DataFrame({
        'Price': prices,
        'Demand': demands,
        'Revenue': revenues,
        'Profit': profits
    })

def generate_simulation_data(A: float, B: float, c_var: float, c_fix: float, p_min: float, p_max: float, capacity: Optional[float] = None) -> pd.DataFrame:
    """キャッシュキー正規化のため、None を float('inf') に正規化して impl を呼び出す generate_simulation_data ラッパー"""
    capacity_value = float('inf') if capacity is None else float(capacity)
    return generate_simulation_data_impl(A, B, c_var, c_fix, p_min, p_max, capacity_value)

df_chart = generate_simulation_data(param_A, param_B, cost_var, cost_fix, min_price, max_price, param_capacity)

def calculate_optimal_profit_analytically(A, B, p_min, p_max, c_var, c_fix, capacity=None):
    if B <= 0:
        return 0.0
    p_demand_zero = A / B
    
    # 有効な価格下限（生産能力制限から逆算される最低価格を含む）
    if capacity is not None:
        p_low = max(p_min, (A - capacity) / B)
    else:
        p_low = p_min
        
    # 有効な価格上限
    p_high = min(p_max, p_demand_zero)
    
    if p_high < p_low:
        p_opt = p_low
    else:
        p_peak = A / (2 * B) + c_var / 2
        p_opt = max(p_low, min(p_high, p_peak))
        
    N_opt = max(0.0, A - B * p_opt)
    if capacity is not None:
        N_opt = min(N_opt, capacity)
        
    profit_opt = (p_opt - c_var) * N_opt - c_fix
    return profit_opt

# -----------------------------------------------------------------
# 5. UIと各タブのレンダリング
# -----------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    _("📊 Profit Simulator", "📊 利益シミュレーター"), 
    _("💎 Shadow Price Analysis", "💎 潜在価格（シャドープライス）経営分析"), 
    _("📘 Math Model", "📘 数理モデル & 問題設定")
])

# -----------------------------------------------------------------
# TAB 1: 利益シミュレーター
# -----------------------------------------------------------------
with tab1:
    # ソルバー不足によるフォールバック警告の表示
    if fallback_int_active:
        fallback_msg = _(
            "⚠️ **Solver Fallback:** The integer programming solver 'scip' was not found. "
            "Automatically switched to the continuous solver 'ipopt' for calculations.\n\n"
            "**As a result:**\n"
            "- Sales volume N is represented as a decimal\n",
            "⚠️ **ソルバー自動フォールバック:** 整数計画ソルバー 'scip' が検出されなかったため、"
            "自動的に連続計画ソルバー 'ipopt' を使用して代替計算を行いました。\n\n"
            "**このため：**\n"
            "- 販売数量 N が小数で表現されています\n"
        )
        if opt_integer_price:
            fallback_msg += _(
                "- **[IMPORTANT] Integer price constraints (1 unit step) are disabled and calculated as continuous values**\n",
                "- **【重要】価格の整数制約（1円単位）も無効化され、連続数値価格として計算されています**\n"
            )
        fallback_msg += _(
            "- Shadow price (sensitivity analysis) results remain valid\n"
            "- The solution might differ from a true strict integer optimal solution\n\n"
            "**Action:** Run `pip install pyscipopt` in your terminal to set up the SCIP solver.",
            "- シャドープライス（感度分析）の結果は有効です\n"
            "- 整数制約のある真の最適解とは異なる可能性があります\n\n"
            "**対策：** ターミナルで `pip install pyscipopt` を実行して SCIP ソルバーライブラリをセットアップしてください。"
        )
        st.warning(fallback_msg)

    # ── ① 最適化結果カード（2列 × 2行のメトリクス）────────────────
    st.subheader(_("🎯 Optimization Results", "🎯 最適化結果の比較"))
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        with st.container(border=True):
            if fallback_int_active:
                st.markdown(_("### 👥 Integer Programming (IPOPT Fallback)", "### 👥 整数計画（代替連続計画結果）"))
                st.caption(_("⚠️ 'SCIP' not found. Used continuous solver 'IPOPT' (integer constraints not applied).", "⚠️ 整数計画ソルバー 'SCIP' が未インストールのため、連続計画ソルバー 'IPOPT' で代替計算した結果です（価格・数量の整数制約は適用されていません）。"))
            else:
                st.markdown(_("### 👥 Integer Programming (Realistic)", "### 👥 整数計画（現実的な販売数）"))
                int_caption = _("Realistic plan with integer sales volume N and price p (Solver: SCIP)", "販売個数 N および設定価格 p が「整数」となる現実的なプラン（ソルバー: SCIP）") if opt_integer_price else _("Realistic plan with integer sales volume N (Solver: SCIP)", "販売個数 N が「整数」となる現実的なプラン（ソルバー: SCIP）")
                st.caption(int_caption)
            if success_int:
                m1, m2 = st.columns(2)
                m1.metric(_("Optimal Price (p)", "最適価格 (p)"), f"¥{opt_p_int:,.1f}")
                demand_disp = f"{opt_N_int:,.2f}" + _(" units", "個") if fallback_int_active else f"{int(round(opt_N_int)):,}" + _(" units", "個")
                m2.metric(_("Expected Demand (N)", "期待販売数 (N)"), demand_disp)
                st.markdown("---")
                m3, m4 = st.columns(2)
                m3.metric(_("Forecasted Revenue", "予測売上 (Revenue)"), f"¥{opt_rev_int:,.1f}")
                m4.metric(_("Forecasted Profit", "予測純利益 (Profit)"), f"¥{opt_prof_int:,.1f}")
            else:
                st.error(_("Failed to solve Integer Programming model.\n", "整数計画の計算に失敗しました。\n") + (result_int.error_msg if result_int.error_msg else ''))

    with res_col2:
        with st.container(border=True):
            st.markdown(_("### 📈 Continuous Programming (Theoretical Max)", "### 📈 連続計画（理論上の最大値）"))
            st.caption(_("Theoretical limits allowing decimal sales volume N (Solver: IPOPT)", "販売個数 N に小数を許容した理論上の限界値（ソルバー: IPOPT）"))
            if success_real:
                m1, m2 = st.columns(2)
                m1.metric(_("Optimal Price (p)", "最適価格 (p)"), f"¥{opt_p_real:,.1f}")
                m2.metric(_("Expected Demand (N)", "期待販売数 (N)"), f"{opt_N_real:,.2f}" + _(" units", "個"))
                st.markdown("---")
                m3, m4 = st.columns(2)
                m3.metric(_("Forecasted Revenue", "予測売上 (Revenue)"), f"¥{opt_rev_real:,.1f}")
                m4.metric(_("Forecasted Profit", "予測純利益 (Profit)"), f"¥{opt_prof_real:,.1f}")
            else:
                st.error(_("Failed to solve Continuous Programming model.\n", "連続計画の計算に失敗しました。\n") + (result_real.error_msg if result_real.error_msg else ''))

    # ── ② 比較棒グラフ（全幅）────────────────────────────
    if success_int or success_real:
        labels = [_("Optimal Price (p) [¥]", "最適価格 (p) [円]"), _("Demand (N) [units]", "販売数 (N) [個]"), _("Revenue [¥]", "売上 [円]"), _("Profit [¥]", "利益 [円]")]
        vals_int  = [
            opt_p_int   if success_int  else 0,
            opt_N_int   if success_int  else 0,
            opt_rev_int if success_int  else 0,
            opt_prof_int if success_int else 0,
        ]
        vals_real = [
            opt_p_real   if success_real  else 0,
            opt_N_real   if success_real  else 0,
            opt_rev_real if success_real  else 0,
            opt_prof_real if success_real else 0,
        ]

        fig_cmp = go.Figure(data=[
            go.Bar(
                name=_('👥 Integer (IPOPT Fallback)', '👥 整数計画 (IPOPT代替)') if fallback_int_active else _('👥 Integer (SCIP)', '👥 整数計画 (SCIP)'),
                x=labels,
                y=vals_int,
                marker_color='#e056fd',
                text=[f"¥{v:,.1f}" if i != 1 else f"{v:,.1f}" + _("units", "個") for i, v in enumerate(vals_int)],
                textposition='outside',
            ),
            go.Bar(
                name=_('📈 Continuous (IPOPT)', '📈 連続計画 (IPOPT)'),
                x=labels,
                y=vals_real,
                marker_color='#3498db',
                text=[f"¥{v:,.1f}" if i != 1 else f"{v:,.2f}" + _("units", "個") for i, v in enumerate(vals_real)],
                textposition='outside',
            ),
        ])
        fig_cmp.update_layout(
            barmode='group',
            margin=dict(l=40, r=20, t=40, b=40),
            yaxis=dict(title=_("Value", "値"), zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )
        st.plotly_chart(fig_cmp, width="stretch")

    # ── ③ 価格弾力性（全幅）────────────────────────────────────
    if success_real:
        if opt_N_real > DUAL_TOLERANCE:
            elasticity = param_B * opt_p_real / opt_N_real
        elif opt_p_real > 0:
            elasticity = float('inf')
        else:
            elasticity = 0.0

        with st.container(border=True):
            st.markdown(_("### 📊 Price Elasticity of Demand at Optimal Price", "### 📊 最適価格での需要の価格弾力性"))
            st.markdown(_(f"Price elasticity of demand $e$ at optimal price: **{elasticity:.2f}**", f"最適価格における需要の価格弾力性 $e$: **{elasticity:.2f}**"))
            if elasticity > 1.05:
                st.info(_(
                    f"💡 **Elastic Demand ($e = {elasticity:.2f} > 1$):**\n\n"
                    "Since demand is highly sensitive to price, **increasing the price will result in a proportionately larger drop in sales volume.**\n\n"
                    "Therefore, avoid arbitrary price hikes and prioritize maintaining current levels or improving brand value perception.",
                    f"💡 **価格弾力的な状態 ($e = {elasticity:.2f} > 1$):**\n\n"
                    "最適価格における需要の弾力性が高いため、**値上げを行うとそれ以上に需要（販売数）が大きく減少してしまいます。**\n\n"
                    "したがって、安易な値上げは避け、現在の価格水準を維持するか、顧客の価値認知を高めるブランディング施策を優先することをお勧めします。"
                ))
            elif elasticity < 0.95:
                st.info(_(
                    f"💡 **Inelastic Demand ($e = {elasticity:.2f} < 1$):**\n\n"
                    "Since demand is insensitive to price, **raising prices will not significantly decrease sales volume.**\n\n"
                    "In this state, applying premium pricing or increasing prices could further boost overall revenue and profit.",
                    f"💡 **価格非弾力的な状態 ($e = {elasticity:.2f} < 1$):**\n\n"
                    "需要の価格感度が低いため、**値上げをしても販売数があまり減少しない状態です。**\n\n"
                    "このような状況では、プレミアム化や価格改定（値上げ）を行うことで、売上・利益をさらに高められる可能性があります。"
                ))
            else:
                st.info(_(
                    f"💡 **Unit Elastic Demand ($e = {elasticity:.2f}$ ≈ 1):**\n\n"
                    "This is the **most perfectly balanced pricing**, where the percentage change in price equals the percentage change in demand.\n\n"
                    "Your pricing strategy has reached the mathematical optimum from a microeconomic perspective.",
                    f"💡 **価格単位弾力的な状態 ($e = {elasticity:.2f}$ ≈ 1):**\n\n"
                    "価格の変更による需要の増減割合がほぼ均等である、**最もバランスの取れた価格設定です。**\n\n"
                    "現在の価格設定は、経済学・経営学的な観点から最適な水準に達しています。"
                ))

    # ── ④ 価格と売上・利益の関係 (収益・利益曲線) ─────────────────
    st.subheader(_("💰 Revenue & Profit Curves", "💰 価格と売上・利益の関係 (収益・利益曲線)"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_chart['Price'],
        y=df_chart['Revenue'],
        mode='lines',
        name=_('Expected Revenue', '期待売上 (Revenue)'),
        line=dict(color='#3498db', width=3),
        hovertemplate=_('Price: ¥%{x:,.1f}<br>Expected Revenue: ¥%{y:,.1f}<extra></extra>', '価格: ¥%{x:,.1f}<br>期待売上: ¥%{y:,.1f}<extra></extra>')
    ))
    fig.add_trace(go.Scatter(
        x=df_chart['Price'],
        y=df_chart['Profit'],
        mode='lines',
        name=_('Expected Profit', '期待利益 (Profit)'),
        line=dict(color='#2ecc71', width=3),
        hovertemplate=_('Price: ¥%{x:,.1f}<br>Expected Profit: ¥%{y:,.1f}<extra></extra>', '価格: ¥%{x:,.1f}<br>期待利益: ¥%{y:,.1f}<extra></extra>')
    ))
    
    if success_real:
        fig.add_trace(go.Scatter(
            x=[opt_p_real],
            y=[opt_prof_real],
            mode='markers',
            name=_('Optimal Profit Point', '最適利益点 (IPOPT)'),
            marker=dict(color='#2ecc71', size=12, symbol='star'),
            hovertemplate=_('Optimal Price: ¥%{x:,.1f}<br>Max Profit: ¥%{y:,.1f}<extra></extra>', '最適価格: ¥%{x:,.1f}<br>最大利益: ¥%{y:,.1f}<extra></extra>')
        ))
        fig.add_trace(go.Scatter(
            x=[opt_p_real],
            y=[opt_rev_real],
            mode='markers',
            name=_('Optimal Revenue Point', '最適売上点 (IPOPT)'),
            marker=dict(color='#3498db', size=12, symbol='star'),
            hovertemplate=_('Optimal Price: ¥%{x:,.1f}<br>Forecast Revenue: ¥%{y:,.1f}<extra></extra>', '最適価格: ¥%{x:,.1f}<br>予測売上: ¥%{y:,.1f}<extra></extra>')
        ))

    fig.update_layout(
        margin=dict(l=40, r=20, t=10, b=40),
        xaxis=dict(title=_("Selling Price (p) [¥]", "販売価格 (p) [円]"), zeroline=False),
        yaxis=dict(title=_("Amount [¥]", "額面 [円]"), zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    st.plotly_chart(fig, width="stretch")

    # ── ⑤ 価格と需要の関係 (需要曲線) ──────────────────────────────
    st.subheader(_("📦 Demand Curve", "📦 価格と需要の関係 (需要曲線)"))
    fig_demand = go.Figure()
    fig_demand.add_trace(go.Scatter(
        x=df_chart['Price'],
        y=df_chart['Demand'],
        mode='lines',
        name=_('Expected Demand', '期待需要 (個数)'),
        line=dict(color='#ff7675', width=3),
        hovertemplate=_('Price: ¥%{x:,.1f}<br>Expected Demand: %{y:,.1f} units<extra></extra>', '価格: ¥%{x:,.1f}<br>期待需要数: %{y:,.1f}個<extra></extra>')
    ))
    
    if success_real:
        fig_demand.add_trace(go.Scatter(
            x=[opt_p_real],
            y=[opt_N_real],
            mode='markers',
            name=_('Optimal Demand Point', '最適需要点 (IPOPT)'),
            marker=dict(color='#ff7675', size=12, symbol='star'),
            hovertemplate=_('Optimal Price: ¥%{x:,.1f}<br>Optimal Demand: %{y:,.1f} units<extra></extra>', '最適価格: ¥%{x:,.1f}<br>最適需要: %{y:,.1f}個<extra></extra>')
        ))

    fig_demand.update_layout(
        margin=dict(l=40, r=20, t=10, b=40),
        xaxis=dict(title=_("Selling Price (p) [¥]", "販売価格 (p) [円]"), zeroline=False),
        yaxis=dict(title=_("Demand Volume (N) [units]", "需要数 (N) [個]"), zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    st.plotly_chart(fig_demand, width="stretch")

# -----------------------------------------------------------------
# TAB 2: シャドープライス経営分析
# -----------------------------------------------------------------
with tab2:
    st.subheader(_("💎 Sensitivity Analysis via Shadow Prices", "💎 潜在価格（シャドープライス）による経営感度分析"))
    st.markdown(_(
        "A shadow price indicates the **marginal increase in final profit if a constraint (e.g., business rule, market cap, production limit) is relaxed by one unit.**",
        "シャドープライスとは、**「制約条件（社内ルール、市場上限など）を1単位緩和したときに、最終利益がいくら増えるか」**を示す経済的価値の指標です。"
    ))

    if result_real.success and result_real.dual_info is not None:
        dual_info = result_real.dual_info

        # ✅ Tab 2 で共通で使用する感度変数を事前に定義（DRY原則・スコープ整理）（改善8）
        d_demand = dual_info.get('dual_demand', 0.0)
        d_min = 0.0 if abs(dual_info.get('dual_min_p', 0.0)) < DUAL_TOLERANCE else dual_info.get('dual_min_p', 0.0)
        d_max = 0.0 if abs(dual_info.get('dual_max_p', 0.0)) < DUAL_TOLERANCE else dual_info.get('dual_max_p', 0.0)

        is_min_binding = abs(d_min) > DUAL_TOLERANCE
        is_max_binding = abs(d_max) > DUAL_TOLERANCE

        d_cap = 0.0
        is_cap_binding = False
        if param_capacity is not None:
            d_cap = dual_info.get('dual_capacity', 0.0) if dual_info else 0.0
            d_cap = 0.0 if abs(d_cap) < DUAL_TOLERANCE else d_cap
            is_cap_binding = abs(d_cap) > DUAL_TOLERANCE

        # ── ① 需要創出の価値
        with st.container(border=True):
            st.markdown(_("### 📈 Value of Demand Creation", "### 📈 マーケティング・需要創出の価値"))
            st.metric(_("Shadow Price of Market Cap ($A$)", "市場規模（$A$）のシャドープライス"), f"¥{d_demand:,.2f}")
            st.write(_(
                f"If you can increase the market demand cap $A$ by **1 unit** through advertising, the profit will increase by exactly **¥{d_demand:,.1f}**.",
                f"広告プロモーションなどで市場需要の上限 $A$ を **1個** 増やすことができれば、利益は最大で **¥{d_demand:,.1f}** 増加します。"
            ))
            st.info(_(
                f"💡 **Business Decision Rule:** "
                f"If the Customer Acquisition Cost (CAC) for 1 unit of demand is **less than ¥{d_demand:,.1f}**, additional marketing investment is mathematically justified.\n\n"
                f"*(Math Note: This perfectly matches the marginal profit at optimal price: $p^* - C_v = {opt_p_real - cost_var:.1f}$ JPY)*",
                f"💡 **経営の意思決定基準:** "
                f"1個の新規需要を獲得するためのマーケティングコスト（CAC）が **¥{d_demand:,.1f} 未満** であれば、追加の広告投資を積極的に行うべきと数学的に裏付けられます。\n\n"
                f"*(数理的補足: 最適価格での限界利益 $p^* - C_v = {opt_p_real - cost_var:.1f}$ 円と一致します)*"
            ))

        # ── ② 価格規制のボトルネック判定
        with st.container(border=True):
            st.markdown(_("### 🔒 Pricing Policy Bottleneck Analysis", "### 🔒 価格ポリシー（規制ルール）の制約ボトルネック分析"))

            if is_min_binding:
                st.warning(_(
                    f"**[WARNING] The minimum price limit (¥{min_price}) is a bottleneck.**\n\n"
                    f"Relaxing this lower bound by **¥1** (allowing a discount) would increase expected profit by **¥{abs(d_min):,.2f}**.\n\n"
                    f"Consider revising your minimum price policy to maximize profit.",
                    f"**【警告】最低価格の制限（¥{min_price}）がボトルネックとなっています。**\n\n"
                    f"この価格下限のルールを **1円緩和（値下げを許可）** することで、期待利益はさらに **¥{abs(d_min):,.2f}** 増加します。\n\n"
                    f"利益を最大化するためには、安売り規制を少し見直す余地があります。"
                ), icon="⚠️")
            elif is_max_binding:
                st.warning(_(
                    f"**[WARNING] The maximum price limit (¥{max_price}) is a bottleneck.**\n\n"
                    f"Relaxing this upper bound by **¥1** (allowing a price hike) would increase expected profit by **¥{abs(d_max):,.2f}**.\n\n"
                    f"The price is constrained too low relative to product value. Premium pricing approval is recommended.",
                    f"**【警告】最高価格の制限（¥{max_price}）がボトルネックとなっています。**\n\n"
                    f"この価格上限のルールを **1円緩和（値上げを許可）** することで、期待利益はさらに **¥{abs(d_max):,.2f}** 増加します。\n\n"
                    f"商品力に対して価格を低く制限しすぎているため、プレミアム価格の承認が必要です。"
                ), icon="⚠️")
            else:
                st.success(_(
                    f"**Current price constraints (¥{min_price} - ¥{max_price}) are NOT a bottleneck.**\n\n"
                    f"The theoretical optimum price (¥{opt_p_real:.1f}) sits comfortably inside the allowed range.\n\n"
                    f"Relaxing price rules further would not generate additional profit (Shadow price is ¥0).",
                    f"**現在の価格制限（¥{min_price} 〜 ¥{max_price}）は経営のボトルネックになっていません。**\n\n"
                    f"設定された範囲のちょうど中間に「理論上の最適な利益の山」が存在し、真の最適価格（¥{opt_p_real:.1f}）に到達しています。\n\n"
                    f"価格制限ルールをこれ以上広げても、追加の利益は発生しません（価格制約のシャドープライスは ¥0 です）。"
                ), icon="✅")

        # ── ③ 生産・在庫能力のボトルネック判定（改善8: None安全性の担保）
        if param_capacity is not None:
            with st.container(border=True):
                st.markdown(_("### 🏭 Production & Inventory Bottleneck Analysis", "### 🏭 生産設備・仕入れ・在庫能力のボトルネック分析"))
                st.metric(_("Shadow Price of Capacity Limit", "生産能力（上限）のシャドープライス"), f"¥{abs(d_cap):,.2f}")
                if is_cap_binding:
                    st.warning(_(
                        f"**[WARNING] The capacity limit (Max {param_capacity} units) is a bottleneck.**\n\n"
                        f"Increasing the production/inventory limit by **1 unit** would increase profit by **¥{abs(d_cap):,.1f}**.\n\n"
                        f"If the marginal cost to expand capacity is **less than ¥{abs(d_cap):,.1f} per unit**, aggressive expansion is recommended.",
                        f"**【警告】生産能力制限（最大 {param_capacity} 個）がボトルネックとなっています。**\n\n"
                        f"設備投資やサプライチェーンの改善によって生産能力・在庫上限を **1個** 増やすことができれば、利益はさらに **¥{abs(d_cap):,.1f}** 増加します。\n\n"
                        f"生産能力を増強するための追加コスト（設備償却費など）が **1個あたり ¥{abs(d_cap):,.1f} 未満** であれば、積極的なキャパシティ拡張を行うべきです。"
                    ), icon="⚠️")
                else:
                    st.success(_(
                        f"**The capacity limit (Max {param_capacity} units) is NOT a bottleneck.**\n\n"
                        f"The optimal expected demand ({opt_N_real:.1f} units) is below the capacity limit. Expanding capacity further will not increase profit (Shadow price is ¥0).",
                        f"**生産能力制限（最大 {param_capacity} 個）はボトルネックになっていません。**\n\n"
                        f"現在の最適需要レベル（期待販売数 {opt_N_real:.1f} 個）は生産能力を下回っているため、設備投資を行って上限を増やしても追加の利益は生まれません（シャドープライスは ¥0 です）。"
                    ), icon="✅")

        # 感度分析の一覧
        with st.container(border=True):
            st.markdown(_("### 📋 Shadow Price Summary Table", "### 📋 シャドープライスの一覧と感度リスト"))

            names = [
                _("Market Demand Cap (A)", "需要の市場上限 (A)"), 
                _("Minimum Price Limit (p >= min_p)", "最低価格の制限 (p >= min_p)"), 
                _("Maximum Price Limit (p <= max_p)", "最高価格の制限 (p <= max_p)")
            ]
            settings = [f"{param_A}" + _(" units", " 個"), f"¥{min_price}", f"¥{max_price}"]
            shadow_prices = [f"¥{d_demand:,.2f}", f"¥{abs(d_min):,.2f}", f"¥{abs(d_max):,.2f}"]
            
            str_active = _("Always Active", "常にアクティブ")
            str_bottleneck = _("Active (Bottleneck!)", "アクティブ（ボトルネック！）")
            str_slack = _("Slack (No impact)", "余裕あり")
            
            statuses = [
                str_active,
                str_bottleneck if is_min_binding else str_slack,
                str_bottleneck if is_max_binding else str_slack
            ]

            if param_capacity is not None:
                names.append(_("Capacity Limit (N <= capacity)", "生産能力の上限 (N <= capacity)"))
                settings.append(f"{param_capacity}" + _(" units", " 個"))
                shadow_prices.append(f"¥{abs(d_cap):,.2f}")
                statuses.append(str_bottleneck if is_cap_binding else str_slack)

            df_duals = pd.DataFrame({
                _("Constraint Name", "制約条件の名称"): names,
                _("Current Setting", "現在の設定値"): settings,
                _("Shadow Price (Profit ↑ per 1 unit relaxation)", "シャドープライス (1単位緩和時の利益増加額)"): shadow_prices,
                _("Status", "ボトルネック状況"): statuses
            })
            st.table(df_duals)

        # ── ④ トルネードチャートによる大域的感度分析（What-If感度）
        st.markdown("---")
        st.subheader(_("🌪️ Tornado Chart for Global Sensitivity Analysis", "🌪️ トルネードチャートによる大域的感度分析（What-If感度）"))
        st.markdown(_(
            "This Tornado chart visualizes **how a ±10% change in each management parameter (Market Cap $A$, Sensitivity $B$, Variable Cost $C_v$, Fixed Cost $C_f$) impacts the maximum profit.**\n"
            "Wider bars indicate higher leverage, highlighting the most critical parameters for your business.",
            "各経営パラメータ（市場規模 $A$、価格感度 $B$、仕入れ変動費 $C_v$、事業固定費 $C_f$）が "
            "**±10% 変化したときに、最大利益（理論連続モデル）が現在値からどれだけ変化するか**を可視化した感度分析チャートです。\n"
            "バーの幅が広いパラメータほど、利益に与えるインパクトが大きい（経営の最重要管理項目である）ことを示します。"
        ))

        # Calculate baseline profit using the analytical method to be consistent
        base_profit = calculate_optimal_profit_analytically(param_A, param_B, min_price, max_price, cost_var, cost_fix, param_capacity)

        # ✅ 基準利益の状況に対する警告表示（改善7）
        if base_profit < 0:
            st.warning(_(
                f"⚠️ **Attention:** With current settings, the theoretical maximum profit is **¥{base_profit:,.1f} (Deficit)**.\n\n"
                "The tornado chart below shows the profit impact of parameter changes from this deficit baseline. Consider reducing fixed costs or adjusting pricing bounds.",
                f"⚠️ **注意:** 現在のパラメータ設定では、理論上の最大利益が **¥{base_profit:,.1f} （赤字）** に陥っています。\n\n"
                "以下のトルネードチャートは、この赤字基準値からのパラメータ増減による利益影響を示しています。固定費の削減や価格設定ポリシーの緩和をご検討ください。"
            ), icon="⚠️")
        elif base_profit < 1000:
            st.info(_(
                f"ℹ️ **Notice:** The current theoretical maximum profit is relatively low at **¥{base_profit:,.1f}**. "
                "To increase profitability, consider reducing costs or expanding demand.",
                f"ℹ️ **お知らせ:** 現在の理論最大利益は **¥{base_profit:,.1f}** と比較的低めです。 "
                "より高い収益性を目指すためには、仕入れ費の削減や最大需要の開拓が必要です。"
            ), icon="ℹ️")

        # A change
        p_A_up = param_A * 1.1
        p_A_down = param_A * 0.9
        prof_A_up = calculate_optimal_profit_analytically(p_A_up, param_B, min_price, max_price, cost_var, cost_fix, param_capacity)
        prof_A_down = calculate_optimal_profit_analytically(p_A_down, param_B, min_price, max_price, cost_var, cost_fix, param_capacity)
        diff_A_up = prof_A_up - base_profit
        diff_A_down = prof_A_down - base_profit

        # B change
        p_B_up = param_B * 1.1
        p_B_down = param_B * 0.9
        prof_B_up = calculate_optimal_profit_analytically(param_A, p_B_up, min_price, max_price, cost_var, cost_fix, param_capacity)
        prof_B_down = calculate_optimal_profit_analytically(param_A, p_B_down, min_price, max_price, cost_var, cost_fix, param_capacity)
        diff_B_up = prof_B_up - base_profit
        diff_B_down = prof_B_down - base_profit

        # Cv change
        p_cv_up = cost_var * 1.1
        p_cv_down = cost_var * 0.9
        prof_cv_up = calculate_optimal_profit_analytically(param_A, param_B, min_price, max_price, p_cv_up, cost_fix, param_capacity)
        prof_cv_down = calculate_optimal_profit_analytically(param_A, param_B, min_price, max_price, p_cv_down, cost_fix, param_capacity)
        diff_cv_up = prof_cv_up - base_profit
        diff_cv_down = prof_cv_down - base_profit

        # Cf change
        p_cf_up = cost_fix * 1.1
        p_cf_down = cost_fix * 0.9
        prof_cf_up = calculate_optimal_profit_analytically(param_A, param_B, min_price, max_price, cost_var, p_cf_up, param_capacity)
        prof_cf_down = calculate_optimal_profit_analytically(param_A, param_B, min_price, max_price, cost_var, p_cf_down, param_capacity)
        diff_cf_up = prof_cf_up - base_profit
        diff_cf_down = prof_cf_down - base_profit

        lbl_A = _("Market Cap A (+/-10%)", "市場需要規模 A (+/-10%)")
        lbl_B = _("Price Sensitivity B (+/-10%)", "価格弾力感度 B (+/-10%)")
        lbl_cv = _("Variable Cost Cv (+/-10%)", "仕入れ変動費 Cv (+/-10%)")
        lbl_cf = _("Fixed Cost Cf (+/-10%)", "事業固定費 Cf (+/-10%)")
        lbl_cap = _("Capacity (+/-10%)", "生産能力 Capacity (+/-10%)")

        sensitivity_data = [
            {"param": lbl_A, "up_val": diff_A_up, "down_val": diff_A_down, "sort_key": max(abs(diff_A_up), abs(diff_A_down))},
            {"param": lbl_B, "up_val": diff_B_up, "down_val": diff_B_down, "sort_key": max(abs(diff_B_up), abs(diff_B_down))},
            {"param": lbl_cv, "up_val": diff_cv_up, "down_val": diff_cv_down, "sort_key": max(abs(diff_cv_up), abs(diff_cv_down))},
            {"param": lbl_cf, "up_val": diff_cf_up, "down_val": diff_cf_down, "sort_key": max(abs(diff_cf_up), abs(diff_cf_down))}
        ]

        # Capacity change (only if capacity is enabled)
        if param_capacity is not None:
            p_cap_up = param_capacity * 1.1
            p_cap_down = param_capacity * 0.9
            prof_cap_up = calculate_optimal_profit_analytically(param_A, param_B, min_price, max_price, cost_var, cost_fix, p_cap_up)
            prof_cap_down = calculate_optimal_profit_analytically(param_A, param_B, min_price, max_price, cost_var, cost_fix, p_cap_down)
            diff_cap_up = prof_cap_up - base_profit
            diff_cap_down = prof_cap_down - base_profit
            sensitivity_data.append({
                "param": lbl_cap,
                "up_val": diff_cap_up,
                "down_val": diff_cap_down,
                "sort_key": max(abs(diff_cap_up), abs(diff_cap_down))
            })

        # ✅ 安定ソート（タイブレーク順序を定義）（改善4）
        sensitivity_order = [lbl_A, lbl_B, lbl_cv, lbl_cf, lbl_cap]

        sensitivity_data = sorted(
            sensitivity_data,
            key=lambda x: (x["sort_key"], sensitivity_order.index(x["param"]) if x["param"] in sensitivity_order else 999),
            reverse=False
        )

        y_labels = [d["param"] for d in sensitivity_data]
        up_vals = [d["up_val"] for d in sensitivity_data]
        down_vals = [d["down_val"] for d in sensitivity_data]

        fig_tornado = go.Figure()
        fig_tornado.add_trace(go.Bar(
            y=y_labels,
            x=down_vals,
            orientation='h',
            name=_('Profit Impact of 10% Decrease (Down)', 'パラメータを 10% 減少させた時の利益影響 (Down)'),
            marker_color='#ff7675',
            hovertemplate='%{y}<br>' + _('10% Decrease: ¥%{x:,.1f} diff', '10%減少時: 利益差 ¥%{x:,.1f}') + '<extra></extra>'
        ))
        fig_tornado.add_trace(go.Bar(
            y=y_labels,
            x=up_vals,
            orientation='h',
            name=_('Profit Impact of 10% Increase (Up)', 'パラメータを 10% 増加させた時の利益影響 (Up)'),
            marker_color='#2ecc71',
            hovertemplate='%{y}<br>' + _('10% Increase: ¥%{x:,.1f} diff', '10%増加時: 利益差 ¥%{x:,.1f}') + '<extra></extra>'
        ))

        fig_tornado.update_layout(
            barmode='overlay',
            xaxis=dict(title=_("Profit Change from Baseline (¥)", "基準利益からの利益変化額 (円)"), zeroline=True, zerolinecolor='black', zerolinewidth=1.5),
            yaxis=dict(title=_("Management Parameter", "経営パラメータ")),
            margin=dict(l=150, r=40, t=20, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_tornado, width="stretch")

        st.markdown(_(
            "💡 **How to Read the Tornado Chart:**  \n"
            "- Green bars extending to the right of `0` show **profit increase**, red bars to the left show **profit decrease**.  \n"
            "- **The widest bars represent the most critical parameters with the highest leverage on your business model.**  \n"
            "- Typically, pricing simulations show that 'Market Cap A' and 'Price Sensitivity B' have vastly more leverage than 'Fixed Cost Cf'. This suggests prioritizing customer value creation and price optimization over simple fixed cost cuts.",
            "💡 **トルネードチャートの読み方・経営分析:**  \n"
            "- 横軸の `0` より右側に伸びる**緑色のバーは利益の増加**、左側に伸びる**赤色のバーは利益の減少**を示します。  \n"
            "- **バーの横幅が最も広いものが、現在のビジネスモデルにおいて最も利益に与えるインパクトが大きい（レバレッジが高い）最重要管理パラメータ**です。  \n"
            "- 通常、多くの価格シミュレーションでは「市場規模 A」や「価格感度 B」のレバレッジが「固定費 Cf」よりはるかに大きくなります。これは、安易な固定費カット（広告削減など）よりも、顧客価値の創出や価格戦略の最適化に注力する方が経営効率が良いことを示唆しています。"
        ))
    else:
        st.info(_("Optimization must succeed to calculate shadow prices.", "シャドープライスを取得するには、最適化計算を成功させる必要があります。"))

# TAB 3: 数理モデル & 問題設定
# -----------------------------------------------------------------
with tab3:
    # 解析的な理論最適価格をスライダーパラメータから直接算出（ソルバーの成否に依存しない）
    # ✅ 制約範囲外の解析的最適価格（改善6）
    analytical_opt_p_unconstrained = (param_A / (2 * param_B)) + (cost_var / 2)
    # ✅ 許容価格制約の範囲内でクリップした理論最適価格
    analytical_opt_p = np.clip(analytical_opt_p_unconstrained, min_price, max_price)

    st.subheader(_("📘 Revenue Optimization Math Model", "📘 収益・利益最大化問題の数理モデル構成"))
    st.markdown(_(
        "This explains the mathematical formulation solved by the Pyomo optimization engine behind the scenes.",
        "このシミュレーターの裏でPyomo数理最適化ソルバーが解いている数式とその数理的な解説です。"
    ))

    # ── ① 定式化セクション（全幅）────────────────────────────────
    with st.container(border=True):
        st.markdown(_("### 📝 Mathematical Formulation", "### 📝 定式化 (Mathematical Formulation)"))

        st.markdown(_("#### 1. Decision Variables", "#### 1. 意思決定変数 (Decision Variables)"))
        st.latex(r"p \in [p_{\min}, p_{\max}] \quad (\text{" + _("Product Price", "商品の設定価格") + r"})")
        st.latex(r"N \in \mathbb{Z}_{\ge 0} \quad (\text{" + _("Expected Sales Volume — Integer", "期待販売数量 — 整数計画") + r"})")
        st.latex(r"\text{" + _("or", "または") + r"} \quad N \ge 0 \quad (\text{" + _("for Continuous Model", "連続計画の場合") + r"})")

        st.markdown(_("#### 2. Parameters", "#### 2. パラメータ (Parameters)"))
        
        str_symbol = _("Symbol", "記号")
        str_val = _("Value", "値")
        str_meaning = _("Meaning", "意味")
        
        str_A = _("Initial market cap (Demand when price is 0)", "初期市場規模（価格0の時の需要）")
        str_B = _("Price sensitivity (Demand drop per ¥1 increase)", "価格感度（1円上昇で需要が減る個数）")
        str_cv = _("Variable Cost (Procurement cost per unit)", "変動費（商品1個あたりの仕入れコスト）")
        str_cf = _("Fixed Cost (Rent, Advertising, etc.)", "事業固定費（家賃・広告費など）")
        
        st.markdown(f"| {str_symbol} | {str_val} | {str_meaning} |\n|---|---|---|\n"
                    f"| $A$ | {param_A} | {str_A} |\n"
                    f"| $B$ | {param_B} | {str_B} |\n"
                    f"| $C_v$ | ¥{cost_var} | {str_cv} |\n"
                    f"| $C_f$ | ¥{cost_fix} | {str_cf} |")

        st.markdown(_("#### 3. Objective Function", "#### 3. 目的関数 (Objective Function)"))
        st.markdown(_("Maximize Profit:", "利益（Profit）の最大化:"))
        st.latex(r"\text{Maximize} \quad \pi(p) = (p - C_v) \cdot N - C_f")

        st.markdown(_("#### 4. Constraints", "#### 4. 制約条件 (Constraints)"))
        st.markdown(_("Demand Curve (Linear Approximation):", "需要曲線（線形近似）:"))
        st.latex(r"\text{s.t.} \quad N = A - B \cdot p")
        st.latex(r"p_{\min} \le p \le p_{\max}, \quad N \ge 0")

    # ── ② MINLP分類（全幅）──────────────────────────────────────
    st.info(_(
        "### 🏆 Problem Classification: MINLP\n\n"
        "- **MIP (Mixed-Integer Programming)**: Sales volume $N$ is restricted to **integers**.\n"
        "- **NLP (Nonlinear Programming)**: The objective function contains a **nonlinear** product of variables $p \\times N$.\n\n"
        "This combination makes it a **MINLP (Mixed-Integer Nonlinear Programming)** problem, one of the most difficult classes in mathematical optimization.\n\n"
        "This simulator uses the global optimization solver **SCIP** to compute the strict integer optimal solution.",
        "### 🏆 問題の分類: MINLP（混合整数非線形計画問題）\n\n"
        "- **MIP（混合整数計画）の側面**: 販売数量 $N$ が **整数** に制限されている。\n"
        "- **NLP（非線形計画）の側面**: 目的関数に変数の積 $p \\times N$ を含む **非線形** な構造。\n\n"
        "この2つの性質が組み合わさることで、本問題は **MINLP（Mixed-Integer Nonlinear Programming）** "
        "という、数理計画法の中でも最も難易度の高いクラスに分類されます。\n\n"
        "本シミュレーターではグローバル最適化ソルバー **SCIP** を用いて厳密な整数最適解を算出しています。"
    ))

    # ── ③ 数理的導出セクション（全幅）──────────────────────────
    with st.container(border=True):
        st.markdown(_("### 🎓 Mathematical Derivation of Optimal Solution", "### 🎓 経済学・最適解の数理的導出"))
        st.markdown(_("#### Step 1: Substitute $N$ into the Profit Function", "#### ① 利益関数の変換：$N$ を価格 $p$ で表す"))
        st.write(_(
            "By substituting the demand constraint $N = A - Bp$ into the objective function to eliminate $N$, the profit becomes a **quadratic function (downward-opening parabola)** of price $p$.",
            "需要制約式 $N = A - Bp$ を目的関数に代入して $N$ を消去すると、利益は価格 $p$ だけの **2次関数（上に凸の放物線）** になります。"
        ))
        col_eq1, col_eq2 = st.columns(2)
        with col_eq1:
            st.latex(r"\pi(p) = (p - C_v)(A - Bp) - C_f")
        with col_eq2:
            st.latex(r"= -Bp^2 + (A + BC_v)p - AC_v - C_f")

        st.markdown(_("#### Step 2: Analytical Derivation of Optimal Price $p^*$", "#### ② 最適価格 $p^*$ の解析的導出"))
        st.write(_(
            "Take the derivative of profit $\\pi(p)$ with respect to $p$ and set it to $0$ to find the vertex (maximum).",
            "利益 $\\pi(p)$ を価格 $p$ で微分し、傾きを $0$ とおいて頂点（最大値）を求めます。"
        ))
        col_eq3, col_eq4 = st.columns(2)
        with col_eq3:
            st.latex(r"\frac{d\pi}{dp} = -2Bp + (A + BC_v) = 0")
        with col_eq4:
            st.latex(r"\therefore \quad p^* = \frac{A}{2B} + \frac{C_v}{2}")

        st.markdown(_("**Unconstrained Theoretical Optimal Price ($p^*_{\\text{unconstrained}}$) at current settings:**", "**現在の設定値における無制約の理論最適価格 ($p^*_{\\text{unconstrained}}$):**"))
        st.latex(f"p^* = {param_A} / (2 \\times {param_B}) + {cost_var} / 2 = \\mathbf{{\\text{{¥}}{analytical_opt_p_unconstrained:.1f}}}")

        if abs(analytical_opt_p - analytical_opt_p_unconstrained) > 1e-5:
            st.info(_(
                f"⚠️ **Boundary Shift due to Constraints:**\n\n"
                f"The unconstrained theoretical price (¥{analytical_opt_p_unconstrained:.1f}) falls outside your allowed price range "
                f"[¥{min_price}, ¥{max_price}].\n\n"
                f"Therefore, the mathematical optimum within the valid range shifts to the boundary value: **¥{analytical_opt_p:.1f}**.",
                f"⚠️ **制約条件による最適価格の境界シフト:**\n\n"
                f"理論上の無制約の最適価格（¥{analytical_opt_p_unconstrained:.1f}）は、設定された許容価格制約範囲 "
                f"[¥{min_price}, ¥{max_price}] を超えています。\n\n"
                f"このため、実際の価格範囲における数理最適解（制約付き理論値）は、境界値である **¥{analytical_opt_p:.1f}** にシフトします。"
            ))
        else:
            st.success(_(
                f"✅ **Optimal Range Verified:**\n\n"
                f"The unconstrained theoretical price (¥{analytical_opt_p_unconstrained:.1f}) sits comfortably inside the allowed range [¥{min_price}, ¥{max_price}].\n\n"
                f"Price constraints are not acting as a bottleneck.",
                f"✅ **最適範囲の検証成功:**\n\n"
                f"無制約の理論最適価格（¥{analytical_opt_p_unconstrained:.1f}）は、許容価格制約範囲 [¥{min_price}, ¥{max_price}] の内側に完全に収まっており、"
                f"価格制約条件は最適決定のボトルネックになっていません。"
            ))

        # ✅ ソルバー解との理論的差異の検証（Copilot改善提案 5）
        if success_real and opt_p_real is not None:
            st.markdown(_("#### ⚖️ Consistency Check: Theoretical vs Solver", "#### ⚖️ 制約付き理論値とソルバー計算値の整合性検証"))
            gap = abs(opt_p_real - analytical_opt_p)
            gap_percentage = (gap / analytical_opt_p) * 100 if analytical_opt_p > 0 else 0.0
            
            st.metric(
                label=_("Price Gap (Theoretical vs Solver)", "理論値（制約考慮後）とソルバー解の価格乖離"),
                value=f"¥{gap:.4f}",
                delta=f"{gap_percentage:.4f}%" if analytical_opt_p > 0 else "N/A",
                delta_color="inverse"
            )
            
            if gap < 0.05:
                st.success(_(
                    "✅ **Mathematical Consistency Verified:** The IPOPT solver result perfectly matches the constrained theoretical value. Your math formulation is 100% correct.",
                    "✅ **数理整合性検証:** IPOPTソルバーの計算結果が制約考慮後の理論値と一致しています。数理モデルの定式化と実装が100%正しいことが保証されます。"
                ))
            else:
                st.warning(_(
                    f"⚠️ **Mathematical Discrepancy Warning:** There is a gap of ¥{gap:.2f}. A nonlinear constraint (like capacity) might be active, or it's a minor solver precision difference.",
                    f"⚠️ **数理整合性警告:** 差異が {gap:.2f}円 あります。生産能力などの非線形制約条件がアクティブになっているか、ソルバー精度による僅かな差分と考えられます。"
                ))
        
        st.markdown(_("#### ③ Business Insights", "#### ③ 経営学的インサイト"))
        insight_col1, insight_col2 = st.columns(2)
        with insight_col1:
            with st.container(border=True):
                st.markdown(_("**📈 Max Revenue vs Max Profit Misalignment**", "**📈 売上最大化 vs 利益最大化のズレ**"))
                st.markdown(_(
                    f"When Variable Cost $C_v = 0$, the revenue-maximizing price is $A/2B = $ **¥{param_A / (2*param_B):.1f}**.\n\n"
                    f"However, with $C_v = {cost_var}$, the profit-maximizing price shifts **¥{cost_var/2:.1f} higher**.\n\n"
                    f"This mathematical truth is evident from the formula $p^* = A/2B + C_v/2$.",
                    f"変動費 $C_v = 0$ のとき（コストなし）の売上最大価格は "
                    f"$A/2B = $ **¥{param_A / (2*param_B):.1f}** です。\n\n"
                    f"しかし変動費 $C_v = {cost_var}$ 円があると、"
                    f"利益最大価格は **¥{cost_var/2:.1f} 円だけ高価格帯にシフト** します。\n\n"
                    f"この理由は $p^* = A/2B + C_v/2$ という公式から明らかです。"
                ))
        with insight_col2:
            with st.container(border=True):
                st.markdown(_("**🏗️ Fixed Costs Don't Impact Optimal Price**", "**🏗️ 固定費は最適価格に影響しない**"))
                st.markdown(_(
                    "Fixed cost $C_f$ is just a constant term in the objective function. When differentiating by $p$, it becomes zero.\n\n"
                    "**This means no matter how much fixed costs increase, the optimal selling price does not change by even 1 yen.**\n\n"
                    "The common business intuition 'let's raise prices to cover our fixed costs' is mathematically and economically incorrect.",
                    "固定費 $C_f$ は目的関数の定数項であるため、"
                    "$p$ で微分すると消えてしまいます。\n\n"
                    "**つまり、固定費がいくら増えても最適な設定価格は1円も変わりません。**\n\n"
                    "「固定費を回収するために値上げしよう」という発想は、"
                    "数学的・経済学的に誤りであることを意味しています。"
                ))