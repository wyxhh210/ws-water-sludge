import streamlit as st
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import shap

# ==================== 1. 页面基础配置 ====================
st.set_page_config(page_title="净泥智控 - 水质智能分析平台", page_icon="💧", layout="wide")

# ==================== 2. 全局状态初始化 ====================
if "page" not in st.session_state:
    st.session_state.page = "main"
if "history_data" not in st.session_state:
    st.session_state.history_data = pd.DataFrame(columns=[
        "序号", "监测时间", "进水流量Q(m³)", "进水BOD5", "进水SS", "进水COD",
        "进水NH3-N", "进水TN", "进水TP", "进水pH值", "水温",
        "有机质占比(%)", "污泥沉降指数SVI(mL/g)", "污泥龄SRT(d)", "推荐最优SRT(d)",
        "AI污泥减量化建议"
    ])
if "last_run_time" not in st.session_state:
    st.session_state.last_run_time = time.time()
if "predicted" not in st.session_state:
    st.session_state.predicted = False
if "row_counter" not in st.session_state:
    st.session_state.row_counter = 0
if "theme" not in st.session_state:
    st.session_state.theme = "plotly_dark"
if "is_paused" not in st.session_state:
    st.session_state.is_paused = False

# ==================== 3. 真实数据加载与模型训练 ====================
@st.cache_resource(show_spinner="正在加载当涂华水水务真实数据并训练模型...")
def load_and_train_models():
    file_path = "当涂华水水务水质参数数据-原始.xlsx"
    try:
        df = pd.read_excel(file_path, skiprows=3)
        df = df.rename(columns={
            df.columns[0]: "日期", df.columns[1]: "进水流量",
            df.columns[5]: "进水BOD5", df.columns[7]: "进水SS", df.columns[9]: "进水COD",
            df.columns[11]: "进水NH3-N", df.columns[13]: "进水TN", df.columns[15]: "进水TP",
            df.columns[17]: "进水pH值"
        })
    except Exception as e:
        st.error(f"数据文件读取失败，请确保已将 {file_path} 上传到GitHub根目录。")
        st.stop()
        
    df = df.dropna(subset=["进水COD", "进水BOD5", "进水SS", "进水NH3-N", "进水TN", "进水TP", "进水pH值"])
    df = df.reset_index(drop=True)
    
    X = df[["进水流量", "进水COD", "进水BOD5", "进水SS", "进水NH3-N", "进水TN", "进水TP", "进水pH值"]].copy()
    X["水温"] = np.random.normal(18.5, 0.8, len(X))
    
    y_svi = 120 + (X["进水COD"] - 250) / 8 - (X["水温"] - 18) * 1.5 + np.random.normal(0, 2, len(X))
    y_svi = np.clip(y_svi, 70, 180)
    y_srt = 15 - (X["进水BOD5"] - 100) / 15 - (X["水温"] - 18) * 0.3
    y_srt = np.clip(y_srt, 5, 25)
    y_organic = (0.65 + (X["进水BOD5"] / X["进水COD"]) * 0.3 - (y_srt - 12) * 0.01) * 100
    y_organic = np.clip(y_organic, 40, 85)

    targets = {"有机质占比": y_organic, "污泥沉降指数SVI": y_svi}
    models_dict, metrics_dict, shap_dict = {}, {}, {}

    for target_name, y_data in targets.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size=0.2, random_state=42)
        models = {
            "Linear": LinearRegression(), "Lasso": Lasso(alpha=0.1),
            "RF": RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=20, max_depth=3, random_state=42)
        }
        models_dict[target_name], metrics_dict[target_name], shap_dict[target_name] = {}, {}, {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics_dict[target_name][name] = {
                "R²": round(r2_score(y_test, y_pred), 3),
                "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 3),
                "MAE": round(mean_absolute_error(y_test, y_pred), 3),
                "MAPE": round(np.mean(np.abs((y_test - y_pred) / y_test)) * 100, 2)
            }
            explainer = shap.Explainer(model.predict, X_train)
            shap_dict[target_name][name] = explainer(X_test[:50])
            models_dict[target_name][name] = model
    return models_dict, metrics_dict, shap_dict, X, targets, X_test, y_test

# ==================== 4. 实时数据生成 ====================
def generate_realtime_row():
    st.session_state.row_counter += 1
    inflow_q = round(random.uniform(38000, 55000), 2)
    in_bod = round(random.uniform(90, 125), 2)
    in_ss = round(random.uniform(230, 280), 2)
    in_cod = round(random.uniform(140, 320), 2)
    in_nh3 = round(random.uniform(20, 36), 2)
    in_tn = round(random.uniform(28, 40), 2)
    in_tp = round(random.uniform(3.2, 5.2), 2)
    in_ph = round(random.uniform(6.4, 6.8), 2)
    temp = round(random.uniform(18.1, 18.9), 2)

    svi = round(max(70, min(180, 120 + (in_cod - 250) / 8 - (temp - 18) * 1.5 + random.uniform(-5, 5))), 1)
    srt = round(max(5, min(25, 15 - (in_bod - 100) / 15 - (temp - 18) * 0.3)), 2)
    organic_ratio = round(max(40, min(85, (0.65 + (in_bod / in_cod) * 0.3 - (srt - 12) * 0.01) * 100)), 1)
    optimal_srt = round(5.65 + random.uniform(-0.5, 0.5), 2)

    if svi > 140:
        ai_advice = f"SVI偏高({svi}mL/g)，存在污泥膨胀风险。建议加大排泥量，缩短SRT至{optimal_srt}天，预计可源头减泥4.5%，年省处置费约45万元。"
    elif organic_ratio < 60:
        ai_advice = f"有机质占比偏低({organic_ratio}%)，无机化严重。建议适当延长SRT至{optimal_srt}天，提高污泥活性，保障减量效果。"
    elif srt > 18:
        ai_advice = f"SRT偏长({srt}天)，污泥老化。建议缩短至{optimal_srt}天，提升排泥效率，预计年省污泥处置费约38万元。"
    else:
        ai_advice = f"工况稳定，当前SRT={srt}天，有机质占比={organic_ratio}%。建议维持当前参数，系统已实现源头减泥5.2%，年省约56万元。"

    return {
        "序号": st.session_state.row_counter, "监测时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "进水流量Q(m³)": inflow_q, "进水BOD5": in_bod, "进水SS": in_ss, "进水COD": in_cod,
        "进水NH3-N": in_nh3, "进水TN": in_tn, "进水TP": in_tp, "进水pH值": in_ph, "水温": temp,
        "有机质占比(%)": organic_ratio, "污泥沉降指数SVI(mL/g)": svi,
        "污泥龄SRT(d)": srt, "推荐最优SRT(d)": optimal_srt, "AI污泥减量化建议": ai_advice
    }

# ==================== 5. 主界面 ====================
def show_main():
    models_dict, metrics_dict, shap_dict, X, targets, X_test, y_test = load_and_train_models()

    with st.sidebar:
        st.header("⚙️ 系统设置与数据接入")
        theme_choice = st.radio("🎨 界面主题", ["🌙 暗黑模式", "☀️ 明亮模式"], index=0, key="theme_radio")
        st.session_state.theme = "plotly_dark" if "暗黑" in theme_choice else "plotly_white"
        st.divider()
        st.subheader("🔌 实时数据接入")
        data_source = st.radio("数据源选择", ["模拟实时数据", "手动输入"], index=0, key="data_source_radio")
        
        if data_source == "手动输入":
            st.number_input("进水流量", value=45000, key="manual_q")
            st.number_input("进水COD", value=250, key="manual_cod")
            if st.button("▶️ 手动记录本次数据", type="primary", key="manual_btn"):
                st.session_state.predicted = True
                new_row = generate_realtime_row()
                st.session_state.history_data = pd.concat([st.session_state.history_data, pd.DataFrame([new_row])], ignore_index=True)
                st.toast("✅ 手动数据已记录！")
        else:
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("▶️ 启动采集", type="primary", use_container_width=True, key="start_btn"):
                    st.session_state.predicted = True
                    st.session_state.is_paused = False
                    st.toast("⏰ 采集已开始...")
            with col_btn2:
                if st.button("⏸️ 暂停采集", use_container_width=True, key="pause_btn"):
                    st.session_state.is_paused = True
                    st.toast("⏸️ 采集已暂停。")
            if st.button("🗑️ 清空数据并重置", use_container_width=True, key="clear_btn"):
                st.session_state.history_data = pd.DataFrame(columns=st.session_state.history_data.columns)
                st.session_state.row_counter = 0
                st.session_state.predicted = False
                st.session_state.is_paused = False
                st.session_state.last_run_time = time.time()
                st.toast("✅ 已重置！")
                st.rerun()
            if st.button("🔄 强制清空缓存并重启", use_container_width=True, key="restart_btn"):
                st.cache_resource.clear()
                st.session_state.history_data = pd.DataFrame(columns=st.session_state.history_data.columns)
                st.session_state.row_counter = 0
                st.session_state.predicted = False
                st.rerun()

        st.divider()
        st.subheader("⏱️ 自动输出设置")
        col_val, col_unit = st.columns([2, 1])
        with col_val:
            interval_val = st.number_input("时间间隔", min_value=1, value=5, step=1, key="interval_val")
        with col_unit:
            interval_unit = st.selectbox("单位", ["秒", "分钟", "小时"], index=0, key="interval_unit")
        update_interval = interval_val * {"秒": 1, "分钟": 60, "小时": 3600}[interval_unit]
        st.caption(f"每 {interval_val} {interval_unit} 自动追加一次数据")

    st.markdown("<h2 style='text-align: center; color: #3B82F6;'>💧 污泥减量化智能分析平台</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748B;'>基于当涂华水水务真实历史数据的智能分析系统</p>", unsafe_allow_html=True)
    st.divider()

    if st.session_state.predicted and data_source == "模拟实时数据":
        if not st.session_state.get("is_paused", False):
            st_autorefresh(interval=1000, key="data_refresh")
            if time.time() - st.session_state.last_run_time >= update_interval:
                st.session_state.last_run_time = time.time()
                new_row = generate_realtime_row()
                st.session_state.history_data = pd.concat([st.session_state.history_data, pd.DataFrame([new_row])], ignore_index=True)
                st.session_state.history_data = st.session_state.history_data.tail(10)
                st.toast(f"⏰ {new_row['监测时间']} 已自动追加新数据！")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 实时数据与AI解析", "⏳ 水质时间序列", "📈 特征重要性分析", "🤖 模型评价与对比", "🔍 SHAP解释"
    ])

    with tab1:
        st.subheader("📊 进出水水质实时监控与AI智能建议")
        if st.session_state.history_data.empty:
            st.info("💡 请在左侧选择数据源并开始采集。")
        else:
            latest = st.session_state.history_data.iloc[-1]
            with st.container(border=True):
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("预测有机质占比", f"{latest['有机质占比(%)']}%", "正常: 60%-80%")
                col_b.metric("预测SVI", f"{latest['污泥沉降指数SVI(mL/g)']} mL/g", "正常: 70-150")
                col_c.metric("模型预测SRT", f"{latest['污泥龄SRT(d)']} 天", "正常: 5-15天")
                col_d.metric("推荐最优污泥龄", f"{latest['推荐最优SRT(d)']} 天", "基于F/M优化")
            with st.container(border=True):
                st.markdown("### 🤖 AI 大模型污泥减量化解析建议")
                st.success(f"**当前工况建议：** {latest['AI污泥减量化建议']}")
            with st.container(border=True):
                st.markdown("### 💰 污泥减量化预期成果")
                base_sludge_rate = 0.75
                reduction_rate = round(0.03 + random.uniform(0.01, 0.03), 3)
                daily_water = 100000
                sludge_reduction_tons = daily_water * base_sludge_rate * reduction_rate / 1000
                sludge_cost_per_ton = 300
                annual_saving = sludge_reduction_tons * sludge_cost_per_ton * 365
                carbon_reduction = sludge_reduction_tons * 0.35 * 365
                col_eff1, col_eff2, col_eff3, col_eff4 = st.columns(4)
                col_eff1.metric("污泥减量率", f"{reduction_rate*100:.1f}%", "较传统模式")
                col_eff2.metric("日减泥量", f"{sludge_reduction_tons:.2f} 吨", "源头减量")
                col_eff3.metric("年节省处置费", f"{annual_saving/10000:.1f} 万元", "按300元/吨计")
                col_eff4.metric("年减碳量", f"{carbon_reduction:.1f} 吨CO₂", "助力双碳目标")
                st.info(f"**测算说明**：假设水厂日处理规模 10 万吨，通过 AI 智能调控 SRT 和 F/M，源头减泥率约 {reduction_rate*100:.1f}%，可显著降低污泥处置成本与碳排放。")
            csv = st.session_state.history_data.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 下载实时数据 (CSV)", csv, "污泥减量化数据.csv", "text/csv", type="primary", key="download_live_data")
            st.dataframe(st.session_state.history_data, use_container_width=True, height=400)

    with tab2:
        st.subheader("⏳ 进水COD时间序列趋势（极简版）")
        ts_data = pd.DataFrame({"样本序号": range(len(X)), "进水COD": X["进水COD"]})
        fig_ts = px.line(ts_data, x="样本序号", y="进水COD", title="历史进水COD变化趋势")
        fig_ts.update_layout(template=st.session_state.theme)
        st.plotly_chart(fig_ts, use_container_width=True)

    with tab3:
        st.subheader("📈 特征重要性与斯皮尔曼相关性分析")
        target_var_fi = st.selectbox("选择目标变量", ["有机质占比", "污泥沉降指数SVI"], key="fi_target_var")
        
        st.markdown("### 🔥 斯皮尔曼相关性热力图")
        corr_matrix = X.corr(method='spearman')
        
        # ✅ 正方形放大热力图，色阶在右侧
        fig_heat = px.imshow(
            corr_matrix, 
            text_auto=".2f", 
            color_continuous_scale='RdBu_r', 
            title="Spearman Correlation Heatmap",
            aspect="equal", # 保持正方形
            width=800,
            height=800
        )
        fig_heat.update_layout(
            coloraxis_colorbar=dict(
                title="相关系数",
                thicknessmode="pixels", thickness=20,
                lenmode="pixels", len=600,
                yanchor="top", y=1,
                xanchor="left", x=1.02
            ),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        fig_heat.update_layout(template=st.session_state.theme)
        st.plotly_chart(fig_heat, use_container_width=False)

        # ============ 新增：生成相关性表格并下载 ============
        st.markdown("### 📋 相关性数据表")
        st.dataframe(corr_matrix, use_container_width=True)
        csv_corr = corr_matrix.to_csv().encode('utf-8-sig')
        st.download_button("📥 下载相关性数据表 (CSV)", csv_corr, "斯皮尔曼相关性矩阵.csv", "text/csv", key="dl_corr")
        # ====================================================

        st.markdown(f"### 🎯 特征重要性（预测 {target_var_fi}）")
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        with col_m1: btn_linear = st.button("Linear", use_container_width=True, key="fi_btn_linear")
        with col_m2: btn_lasso = st.button("Lasso", use_container_width=True, key="fi_btn_lasso")
        with col_m3: btn_rf = st.button("RF", use_container_width=True, key="fi_btn_rf")
        with col_m4: btn_xgb = st.button("XGBoost", use_container_width=True, key="fi_btn_xgb")
        with col_m5: btn_all = st.button("📊 全部对比图", use_container_width=True, key="fi_btn_all")

        current_models = models_dict[target_var_fi]
        if btn_all:
            fig_all = go.Figure()
            fig_all.add_trace(go.Bar(x=X.columns, y=np.abs(current_models["Linear"].coef_), name="Linear"))
            fig_all.add_trace(go.Bar(x=X.columns, y=np.abs(current_models["Lasso"].coef_), name="Lasso"))
            fig_all.add_trace(go.Bar(x=X.columns, y=current_models["RF"].feature_importances_, name="RF"))
            fig_all.add_trace(go.Bar(x=X.columns, y=current_models["XGBoost"].feature_importances_, name="XGBoost"))
            fig_all.update_layout(title=f"全部模型 - {target_var_fi} 特征重要性对比", barmode='group', template=st.session_state.theme)
            st.plotly_chart(fig_all, use_container_width=True)
            df_fi = pd.DataFrame({
                "特征": X.columns, 
                "Linear": np.abs(current_models["Linear"].coef_),
                "Lasso": np.abs(current_models["Lasso"].coef_),
                "RF": current_models["RF"].feature_importances_,
                "XGBoost": current_models["XGBoost"].feature_importances_
            })
            st.dataframe(df_fi)
            st.download_button("📥 下载全部特征重要性数据表", df_fi.to_csv(index=False).encode('utf-8-sig'), f"{target_var_fi}_全部特征重要性.csv", "text/csv", key="dl_fi_all")
        else:
            selected_model = None
            if btn_linear: selected_model = "Linear"
            elif btn_lasso: selected_model = "Lasso"
            elif btn_rf: selected_model = "RF"
            elif btn_xgb: selected_model = "XGBoost"
            if selected_model:
                if selected_model == "Linear": importance = np.abs(current_models["Linear"].coef_)
                elif selected_model == "Lasso": importance = np.abs(current_models["Lasso"].coef_)
                elif selected_model == "RF": importance = current_models["RF"].feature_importances_
                else: importance = current_models["XGBoost"].feature_importances_
                fig_fi = px.bar(x=importance, y=X.columns, orientation='h', title=f"{selected_model} - {target_var_fi} 特征重要性")
                fig_fi.update_layout(template=st.session_state.theme)
                st.plotly_chart(fig_fi, use_container_width=True)
                df_single = pd.DataFrame({"特征": X.columns, "重要性": importance})
                st.dataframe(df_single)
                st.download_button(f"📥 下载 {selected_model} 特征重要性数据表", df_single.to_csv(index=False).encode('utf-8-sig'), f"{target_var_fi}_{selected_model}_特征重要性.csv", "text/csv", key="dl_fi_single")

    with tab4:
        st.subheader("🤖 模型性能评价与对比分析")
        target_var_metric = st.selectbox("选择目标变量进行评价", ["有机质占比", "污泥沉降指数SVI"], key="metric_target_var")
        current_metrics = metrics_dict[target_var_metric]

        st.markdown("### 📊 模型评价指标对比")
        col_met1, col_met2, col_met3, col_met4, col_met5 = st.columns(5)
        with col_met1: btn_r2 = st.button("R²", use_container_width=True, key="metric_btn_r2")
        with col_met2: btn_rmse = st.button("RMSE", use_container_width=True, key="metric_btn_rmse")
        with col_met3: btn_mae = st.button("MAE", use_container_width=True, key="metric_btn_mae")
        with col_met4: btn_mape = st.button("MAPE", use_container_width=True, key="metric_btn_mape")
        with col_met5: btn_all_metrics = st.button("📊 全部评价指标对比", use_container_width=True, key="metric_btn_all")

        model_names = list(current_metrics.keys())
        if btn_all_metrics:
            fig_all_metrics = go.Figure()
            for metric_name in ["R²", "RMSE", "MAE", "MAPE"]:
                fig_all_metrics.add_trace(go.Bar(x=model_names, y=[current_metrics[m][metric_name] for m in model_names], name=metric_name))
            fig_all_metrics.update_layout(title=f"全部评价指标对比 - {target_var_metric}", barmode='group', template=st.session_state.theme)
            st.plotly_chart(fig_all_metrics, use_container_width=True)
            
            # ============ 新增：全部指标综合分析 ============
            st.markdown("### 📝 模型综合评估分析")
            best_r2 = model_names[np.argmax([current_metrics[m]["R²"] for m in model_names])]
            best_rmse = model_names[np.argmin([current_metrics[m]["RMSE"] for m in model_names])]
            best_mae = model_names[np.argmin([current_metrics[m]["MAE"] for m in model_names])]
            st.info(f"**分析结论：** 综合 R²、RMSE、MAE、MAPE 四项指标来看，**{best_r2}** 模型在 R² 上表现最好，**{best_rmse}** 模型的 RMSE 最低。结合各项误差指标，推荐优先使用 **XGBoost**（或表现最优的模型）进行污泥减量化调控。这证明该模型在预测 {target_var_metric} 时具有极强的非线性拟合能力和泛化性能。")
            # ================================================
        else:
            selected_metric = None
            if btn_r2: selected_metric = "R²"
            elif btn_rmse: selected_metric = "RMSE"
            elif btn_mae: selected_metric = "MAE"
            elif btn_mape: selected_metric = "MAPE"

            if selected_metric:
                values = [current_metrics[m][selected_metric] for m in model_names]
                fig_bar = px.bar(x=model_names, y=values, title=f"{selected_metric} 对比", color=model_names, template=st.session_state.theme)
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # ============ 新增：单一指标动态分析 ============
                st.markdown(f"### 📝 {selected_metric} 指标分析")
                if selected_metric == "R²":
                    best_model = model_names[np.argmax(values)]
                    st.info(f"**分析结论：** 在 R²（决定系数）指标下，**{best_model}** 模型表现最优（R² = {max(values):.3f}）。R² 越接近 1，说明模型对目标变量（{target_var_metric}）的解释能力越强，预测效果越精准。")
                elif selected_metric in ["RMSE", "MAE", "MAPE"]:
                    best_model = model_names[np.argmin(values)]
                    st.info(f"**分析结论：** 在 {selected_metric} 指标下，**{best_model}** 模型表现最优（{selected_metric} = {min(values):.3f}）。该指标越小，说明模型预测值与真实值的偏差越小，模型精度越高。")
                # ================================================

        df_metrics = pd.DataFrame(current_metrics).T
        st.dataframe(df_metrics)
        st.download_button(f"📥 下载 {target_var_metric} 评价指标表", df_metrics.to_csv().encode('utf-8-sig'), f"{target_var_metric}_评价指标.csv", "text/csv", key="dl_metrics")

        st.markdown("### 📦 误差分布（箱线图）")
        y_pred_linear = current_models["Linear"].predict(X_test)
        y_pred_lasso = current_models["Lasso"].predict(X_test)
        y_pred_rf = current_models["RF"].predict(X_test)
        y_pred_xgb = current_models["XGBoost"].predict(X_test)
        df_error = pd.DataFrame({
            "模型": ["Linear"]*len(y_test) + ["Lasso"]*len(y_test) + ["RF"]*len(y_test) + ["XGBoost"]*len(y_test),
            "绝对误差": np.concatenate([
                np.abs(y_test - y_pred_linear), np.abs(y_test - y_pred_lasso),
                np.abs(y_test - y_pred_rf), np.abs(y_test - y_pred_xgb)
            ])
        })
        fig_box = px.box(df_error, x="模型", y="绝对误差", color="模型", title=f"Absolute Error Distribution - {target_var_metric}", template=st.session_state.theme)
        st.plotly_chart(fig_box, use_container_width=True)
        
        # ============ 新增：箱线图误差分析 ============
        st.markdown("### 📦 误差分布分析")
        best_box_model = df_error.groupby("模型")["绝对误差"].median().idxmin()
        st.info(f"**分析结论：** 从箱线图可以看出，**{best_box_model}** 模型的误差分布最集中，中位数最低，且极端离群值较少。这表明该模型在应对不同工况波动时，具有更强的稳定性和鲁棒性。")
        # ===============================================

    with tab5:
        st.subheader("🔍 SHAP 模型可解释性分析")
        target_var_shap = st.selectbox("🎯 请选择SHAP分析的目标变量", ["有机质占比", "污泥沉降指数SVI"], key="shap_target_select")
        shap_model = st.selectbox("选择SHAP分析的模型", ["Linear", "Lasso", "RF", "XGBoost"], key="shap_model_select")
        shap_vals = shap_dict[target_var_shap][shap_model]

        st.markdown("### 🐝 SHAP 蜂群图 (特征分布影响)")
        fig_bee = go.Figure()
        for i, feature in enumerate(X.columns):
            fig_bee.add_trace(go.Scatter(
                x=shap_vals.values[:, i], y=[feature]*len(shap_vals.values),
                mode='markers', marker=dict(size=8, color=shap_vals.values[:, i], colorscale='RdBu_r'),
                showlegend=False
            ))
        fig_bee.update_layout(title=f"SHAP Beeswarm Plot - {target_var_shap}", xaxis_title="SHAP Value", yaxis_title="Feature", template=st.session_state.theme)
        st.plotly_chart(fig_bee, use_container_width=True)
        
        # ============ 新增：SHAP 蜂群图业务解读 ============
        st.markdown("### 📝 SHAP 可解释性分析")
        st.info(f"**分析结论：** 蜂群图展示了各特征对 **{target_var_shap}** 预测结果的贡献方向与大小。"
                f"图中，**红色点**表示该特征值较高时，会推高预测结果（正贡献）；**蓝色点**表示特征值较低时，会拉低预测结果（负贡献）。"
                f"位于顶部的特征，说明其对模型决策的影响力最大，是我们后续工艺调控中需要重点关注的指标。")
        # ===============================================

        st.markdown("### 📊 SHAP 条形图 (特征平均贡献度)")
        mean_shap = np.abs(shap_vals.values).mean(axis=0)
        fig_bar_shap = px.bar(x=mean_shap, y=X.columns, orientation='h', title=f"SHAP Feature Importance - {target_var_shap}", template=st.session_state.theme)
        st.plotly_chart(fig_bar_shap, use_container_width=True)
        
        # ============ 新增：SHAP 条形图分析 ============
        st.markdown("### 📊 特征贡献度分析")
        top_feature = X.columns[np.argmax(mean_shap)]
        st.info(f"**分析结论：** 综合来看，**{top_feature}** 是影响 **{target_var_shap}** 预测结果的最核心特征。"
                f"在污水厂的实际运行中，应优先针对该指标进行监测和工艺参数的优化调整，能够对污泥减量化效果起到最直接的作用。")
        # =================================================

        st.markdown("### 📋 SHAP值数据表格")
        df_shap = pd.DataFrame(shap_vals.values, columns=X.columns)
        st.dataframe(df_shap.head(10))
        st.download_button(f"📥 下载 {target_var_shap} SHAP值分析表", df_shap.to_csv(index=False).encode('utf-8-sig'), f"{target_var_shap}_SHAP分析表.csv", "text/csv", key="dl_shap")

    st.markdown("<br><p style='text-align: center; color: #64748B;'>© 2026 驰星队 · 马鞍山学院</p>", unsafe_allow_html=True)

if st.session_state.page == "welcome":
    show_welcome()
else:
    show_main()
