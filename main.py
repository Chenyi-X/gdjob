import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
import time

# ==========================================
# 0. 配置与工具函数
# ==========================================
st.set_page_config(page_title="广东省岗位智能推荐系统 Pro", layout="wide", page_icon="💼")

# 预定义的分类关键词映射库
JOB_CATEGORY_KEYWORDS = {
    "事务/职能类 (行政、人事、助理)": ["行政", "人事", "HR", "助理", "文员", "秘书", "前台", "专员", "后勤", "档案"],
    "沟通/销售类 (销售、咨询、客服)": ["销售", "顾问", "业务", "客服", "客户", "经理", "招商", "代表", "置业", "经纪人"],
    "技术/研发类 (开发、运维、工程)": ["工程师", "开发", "运维", "数据", "算法", "IT", "测试", "架构", "前端", "后端"],
    "设计/创意类 (UI、设计、媒体)": ["设计", "UI", "美工", "剪辑", "策划", "文案", "新媒体", "视频", "创意"],
    "财务/金融类 (会计、审计、风控)": ["会计", "财务", "审计", "出纳", "结算", "风控", "投资", "分析师"],
    "运营/管理类 (运营、项目、管培)": ["运营", "项目", "管培生", "储备", "主管", "店长", "调度"],
    "教育/服务类 (教师、培训、服务)": ["教师", "培训", "教务", "服务员", "司机", "保安", "厨师"]
}

# 常见行业列表
INDUSTRY_LIST = [
    "互联网/计算机/软件", "金融/银行/保险", "教育/培训/院校", "房地产/建筑/建材",
    "批发/零售/贸易", "制造业/机械/电子", "医疗/卫生/制药", "物流/运输/仓储",
    "广告/传媒/文化", "政府/公共事业/非盈利", "服务业 (餐饮/酒店/旅游)"
]

# 广东省主要城市列表
GUANGDONG_CITIES = [
    "广州市", "深圳市", "珠海市", "佛山市", "东莞市", "惠州市", "中山市",
    "江门市", "肇庆市", "汕头市", "湛江市", "茂名市", "清远市", "韶关市"
]


# ==========================================
# 1. 核心逻辑层 (已修复学历过滤)
# ==========================================

class JobRecommender:
    def __init__(self, df):
        self.df = df.copy()
        # 数据清洗
        self.df['薪资下限'] = pd.to_numeric(self.df['薪资下限'], errors='coerce').fillna(0)
        self.df['薪资上限'] = pd.to_numeric(self.df['薪资上限'], errors='coerce').fillna(0)
        self.df['平均薪资'] = (self.df['薪资下限'] + self.df['薪资上限']) / 2
        # 填充缺失值
        str_cols = ['职位名称', '行业', '工作地区', '单位名称', '单位性质', '学历要求', '经验要求']
        for col in str_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('')
            else:
                self.df[col] = ''

    def calculate_scores(self, user_profile, weights):
        df = self.df.copy()

        # --- 🔧 核心修复：学历硬性门槛过滤 ---
        # 逻辑：如果岗位要求的学历 > 用户的学历，直接剔除，不予推荐。
        edu_map = {
            '博士': 6,
            '硕士研究生': 5, '硕士': 5,
            '大学本科': 4, '本科': 4,
            '大学专科': 3, '专科': 3,
            '中专': 2, '高中': 2, '中技': 2,
            '初中': 1, '不限': 0, '': 0
        }
        user_edu_val = edu_map.get(user_profile['education'], 3)  # 默认为大专/本科水平

        # 计算每一行的岗位学历值
        def get_job_edu_val(text):
            # 提取学历关键词，例如 "本科/硕士" 取 "本科"
            # 默认给0 (不限)，保证低门槛岗位能通过
            first_req = str(text).split('/')[0]
            return edu_map.get(first_req, 0)

        df['学历数值'] = df['学历要求'].apply(get_job_edu_val)

        # 🚨 执行硬过滤：保留 (岗位要求 <= 用户学历) 的岗位
        # 例如：用户是本科(4)，可以看本科(4)、专科(3)、不限(0)；不能看硕士(5)
        df = df[df['学历数值'] <= user_edu_val].copy()

        # --- 维度 1: 学历评分 (过滤后剩下的都是合格的，但匹配度不同) ---
        def score_edu(job_val):
            # 刚好匹配给100，用户学历远高于岗位给80 (向下兼容)
            if user_edu_val == job_val: return 100
            return 85  # 向下兼容，比如本科生去面专科岗，也是有竞争力的

        df['S_学历'] = df['学历数值'].apply(score_edu)

        # --- 维度 2: 经验匹配 ---
        def score_exp(job_exp):
            job_exp_str = str(job_exp)
            if "无" in job_exp_str or "不限" in job_exp_str: return 100
            if user_profile['experience'] == "应届生":
                return 100 if "应届" in job_exp_str else 60
            return 90 if "应届" not in job_exp_str else 70

        df['S_经验'] = df['经验要求'].apply(score_exp)

        # --- 维度 3: 专业与职能契合度 ---
        target_keywords = JOB_CATEGORY_KEYWORDS.get(user_profile['job_category'], [])
        preferred_industries = user_profile['preferred_industries']

        def score_professional(row):
            score = 0
            title = str(row['职位名称'])
            industry = str(row['行业'])
            for kw in target_keywords:
                if kw in title:
                    score += 50
                    break
            for ind in preferred_industries:
                if ind[:2] in industry or industry in ind:
                    score += 30
                    break
            if user_profile['major'] in title or user_profile['major'] in industry:
                score += 20
            return min(score, 100)

        df['S_专业'] = df.apply(score_professional, axis=1)

        # --- 维度 4: 薪资竞争力 ---
        min_expect = user_profile['min_salary']
        df['S_薪资'] = df['平均薪资'].apply(lambda x: min(120, (x / min_expect * 100)) if x >= min_expect * 0.9 else 40)

        # --- 维度 5: 城市与通勤 ---
        user_cities = user_profile['preferred_cities']
        user_district = user_profile.get('district', '')

        def score_city_location(row):
            loc = str(row['工作地区'])
            score = 40
            if len(user_cities) > 0 and user_cities[0] in loc:
                score = 100
            elif len(user_cities) > 1 and user_cities[1] in loc:
                score = 90
            elif len(user_cities) > 2 and user_cities[2] in loc:
                score = 85
            elif any(c in loc for c in ['广州', '深圳']):
                score = 60

            if user_district and user_district in loc:
                score += 20
            return min(score, 120)

        df['S_城市'] = df.apply(score_city_location, axis=1)

        # --- 维度 6: 稳定性 ---
        stable_keywords = ['国企', '央企', '事业单位', '机关', '学校', '医院', '银行', '分行', '政府', '公办']

        def score_stability(row):
            text = str(row['单位名称']) + str(row['单位性质'])
            for kw in stable_keywords:
                if kw in text: return 100
            return 60

        df['S_稳定'] = df.apply(score_stability, axis=1)

        # --- 维度 7: 潜力 ---
        growth_keywords = ['管培', '储备', '晋升', '培训', '核心', '梯队', '主管']

        def score_growth(text):
            score = 60
            for kw in growth_keywords:
                if kw in str(text): score += 15
            return min(score, 100)

        df['S_潜力'] = df['职位名称'].apply(score_growth)

        # --- 综合加权 ---
        df['综合得分'] = (
                                 df['S_学历'] * weights['学历'] +
                                 df['S_经验'] * weights['经验'] +
                                 df['S_专业'] * weights['专业'] +
                                 df['S_薪资'] * weights['薪资'] +
                                 df['S_城市'] * weights['城市'] +
                                 df['S_潜力'] * weights['潜力'] +
                                 df['S_稳定'] * weights['稳定']
                         ) / 100

        # --- 生成推荐理由 ---
        def get_reason(row):
            tags = []
            if user_district and user_district in str(row['工作地区']):
                tags.append(f"🏠 离家近({user_district})")
            elif row['S_城市'] >= 90:
                tags.append("📍 城市匹配")
            if row['S_薪资'] >= 110: tags.append("💰 薪资优厚")
            if row['S_稳定'] >= 90: tags.append("🛡️ 铁饭碗/稳定")
            if row['S_潜力'] >= 80: tags.append("📈 发展空间大")
            if row['S_专业'] >= 80: tags.append("🎯 专业对口")
            return " | ".join(tags) if tags else "✅ 综合条件匹配"

        df['推荐理由'] = df.apply(get_reason, axis=1)

        # 返回结果 (只要有分数的都返回，筛选在Step4做)
        return df.sort_values(by='综合得分', ascending=False)


# ==========================================
# 2. 交互层
# ==========================================

if 'step' not in st.session_state: st.session_state.step = 1
if 'user_data' not in st.session_state: st.session_state.user_data = {}


@st.cache_data
def load_data(file):
    if file is not None:
        try:
            return pd.read_csv(file)
        except Exception as e:
            st.error(f"文件读取错误: {e}")
            return pd.DataFrame()
    return None


# --- 侧边栏 ---
with st.sidebar:
    st.header("📂 数据导入")
    uploaded_file = st.file_uploader("上传岗位CSV (6w+数据)", type=['csv'])
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.success("数据已就绪")
    st.divider()
    st.info("💡 提示：本系统已升级算法，支持商圈匹配与稳定性识别。")

# --- 进度条 ---
current_step = st.session_state.step
st.progress((current_step - 1) / 3)

# ==========================================
# STEP 1: 基础门槛
# ==========================================
if current_step == 1:
    st.subheader("📝 第一步：基础背景调查")
    col1, col2 = st.columns(2)
    with col1:
        edu = st.selectbox("最高学历", ["博士", "硕士", "大学本科", "大学专科", "中专/高中"], index=2)
        major = st.text_input("主修专业", value="工商管理")
    with col2:
        exp = st.selectbox("工作经验", ["应届生", "1-3年", "3-5年", "5-10年", "10年以上"])
        min_salary = st.number_input("期望最低月薪 (元)", min_value=1000, value=5000, step=500)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("下一步 ➡️", type="primary"):
        st.session_state.user_data.update(
            {'education': edu, 'major': major, 'experience': exp, 'min_salary': min_salary})
        st.session_state.step = 2
        st.rerun()

# ==========================================
# STEP 2: 详细偏好
# ==========================================
elif current_step == 2:
    st.subheader("🏙️ 第二步：工作偏好定制")
    st.markdown("##### 1. 期望工作城市")
    col_city, col_dist = st.columns([2, 1])
    with col_city:
        selected_cities = st.multiselect("按优先级选择 (最多3个)", GUANGDONG_CITIES, max_selections=3)
    with col_dist:
        district = st.text_input("🏠 居住区域/偏好商圈 (选填)", help="例如：天河、南山")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        job_category = st.selectbox("倾向职能", list(JOB_CATEGORY_KEYWORDS.keys()))
    with col2:
        selected_industries = st.multiselect("感兴趣行业", INDUSTRY_LIST, default=["互联网/计算机/软件"])

    st.markdown("<br>", unsafe_allow_html=True)
    col_back, col_next = st.columns([1, 5])
    if col_back.button("⬅️ 上一步"):
        st.session_state.step = 1
        st.rerun()
    if col_next.button("下一步 ➡️", type="primary"):
        if not selected_cities:
            st.error("请至少选择一个城市！")
        else:
            st.session_state.user_data.update(
                {'preferred_cities': selected_cities, 'district': district, 'job_category': job_category,
                 'preferred_industries': selected_industries})
            st.session_state.step = 3
            st.rerun()

# ==========================================
# STEP 3: 价值观与权重 (含缓冲动画)
# ==========================================
elif current_step == 3:
    st.subheader("⚖️ 第三步：职业价值观微调")
    q1 = st.radio("Q1. 面对一份工作，您更看重的是？",
                  ("💰 薪资回报 (钱给够就行)", "📈 成长空间 (接受起薪低但天花板高)", "🏠 稳定与生活 (离家近/铁饭碗)"),
                  horizontal=True)
    q2 = st.slider("Q2. 对“专业对口”的执念？", 0, 100, 50)

    st.markdown("<br>", unsafe_allow_html=True)
    col_back, col_next = st.columns([1, 5])
    if col_back.button("⬅️ 上一步"):
        st.session_state.step = 2
        st.rerun()

    if col_next.button("🚀 生成智能推荐报告", type="primary"):
        weights = {'学历': 10, '经验': 10, '专业': 15, '薪资': 20, '城市': 25, '潜力': 15, '稳定': 5}
        if "薪资" in q1:
            weights['薪资'] += 20;
            weights['城市'] -= 10;
            weights['稳定'] -= 5;
            weights['潜力'] -= 5
        elif "成长" in q1:
            weights['潜力'] += 20;
            weights['薪资'] -= 5;
            weights['稳定'] -= 5;
            weights['学历'] -= 10
        elif "稳定" in q1:
            weights['城市'] += 10;
            weights['稳定'] += 20;
            weights['薪资'] -= 10;
            weights['潜力'] -= 10;
            weights['经验'] -= 10
        weights['专业'] = int(10 + (q2 / 100 * 15))

        # 缓冲动画
        placeholder = st.empty()
        with placeholder.container():
            st.markdown("### 🤖 AI 正在全力计算中...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("🔍 正在解析您的简历画像与偏好...")
            for i in range(30): time.sleep(0.01); progress_bar.progress(i)
            status_text.text(f"🏙️ 正在比对 {len(GUANGDONG_CITIES)} 个城市与商圈距离...")
            for i in range(30, 70): time.sleep(0.01); progress_bar.progress(i)
            status_text.text("🧮 正在执行严格学历过滤与加权评分...")
            for i in range(70, 100): time.sleep(0.01); progress_bar.progress(i)
            time.sleep(0.3)

        st.session_state.weights = weights
        st.session_state.step = 4
        st.rerun()

# ==========================================
# STEP 4: 结果展示 (Tab分层 + 行业薪资透视)
# ==========================================
elif current_step == 4:
    st.balloons()

    # --- 数据加载 ---
    file_obj = st.session_state.get('uploaded_file')
    df = load_data(file_obj)

    if df is None or df.empty:
        st.warning("⚠️ 未检测到上传文件，请在侧边栏重新上传 CSV 文件。")
        st.stop()

    recommender = JobRecommender(df)
    results = recommender.calculate_scores(st.session_state.user_data, st.session_state.weights)

    # 筛选逻辑
    high_score_jobs = results[results['综合得分'] >= 80]
    top_jobs = high_score_jobs if not high_score_jobs.empty else results.head(20)
    top_jobs = top_jobs.sort_values(by='综合得分', ascending=False)


    # --- 0. 智能地址清洗逻辑 ---
    def smart_location_name(loc_str):
        loc_str = str(loc_str).replace("广东省", "")
        user_cities = st.session_state.user_data.get('preferred_cities', [])
        for city in user_cities:
            if city in loc_str:
                loc_str = loc_str.replace(city, "")
        return loc_str if loc_str.strip() else "市辖区/全城"


    top_jobs['显示区域'] = top_jobs['工作地区'].apply(smart_location_name)

    # --- 1. 宏观统计看板 (始终显示) ---
    st.subheader("📊 岗位透视驾驶舱")

    avg_salary = top_jobs['平均薪资'].mean()
    top_area_count = top_jobs['显示区域'].value_counts()
    top_area_name = top_area_count.idxmax()
    top_industry_count = top_jobs['行业'].value_counts()
    top_industry_name = top_industry_count.idxmax() if not top_industry_count.empty else "通用"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("精选岗位数", f"{len(top_jobs)} 个", "综合评分Top序列")
    m2.metric("平均月薪", f"{avg_salary / 1000:.1f} k", help="基于筛选出的岗位平均值")
    m3.metric("热点区域", top_area_name, f"该区占比 {top_area_count.max() / len(top_jobs):.0%}")
    m4.metric("核心行业", top_industry_name, "占比最高的主流行业")

    st.markdown("---")

    # --- 2. Tab 分层视图 ---
    tab_charts, tab_list = st.tabs(["📈 全局透视分析 (决策辅助)", "📋 详细岗位列表 (投递清单)"])

    # === TAB 1: 图表分析 ===
    with tab_charts:
        # 第一行：区域分布 & 薪资分布
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("##### 📍 机会都在哪里？ (区域分布)")
            area_df = top_jobs['显示区域'].value_counts().head(8).sort_values(ascending=True)
            fig_area = go.Figure(go.Bar(
                y=area_df.index, x=area_df.values, orientation='h',
                text=area_df.values, textposition='auto', marker_color='#4F81BD'
            ))
            fig_area.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=250, xaxis_title="岗位数量",
                                   template="plotly_white")
            st.plotly_chart(fig_area, use_container_width=True)

        with c2:
            st.markdown("##### 💰 整体薪资段位分布")
            salary_bins = [0, 5000, 8000, 12000, 20000, 100000]
            salary_labels = ['5k以下', '5k-8k', '8k-12k', '12k-20k', '20k以上']
            top_jobs['薪资段'] = pd.cut(top_jobs['平均薪资'], bins=salary_bins, labels=salary_labels)
            sal_counts = top_jobs['薪资段'].value_counts().sort_index()

            fig_sal = go.Figure(go.Bar(
                x=sal_counts.index, y=sal_counts.values,
                text=sal_counts.values, textposition='auto', marker_color='#C0504D'
            ))
            fig_sal.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=250, yaxis_title="岗位数量",
                                  template="plotly_white")
            st.plotly_chart(fig_sal, use_container_width=True)

        st.divider()

        # 第二行：行业占比 & 行业平均薪资
        c3, c4 = st.columns(2)

        with c3:
            st.markdown("##### 🏭 都是哪些行业的岗位？ (Top 8)")
            # 行业饼图
            ind_counts = top_jobs['行业'].value_counts().head(8)
            fig_pie = go.Figure(data=[go.Pie(
                labels=ind_counts.index, values=ind_counts.values, hole=.4, textinfo='label+percent'
            )])
            fig_pie.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300, showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)

        with c4:
            st.markdown("##### 💵 各行业平均薪资对比 (Top 8)")
            # 计算各行业平均薪资
            ind_salary = top_jobs.groupby('行业')['平均薪资'].mean().sort_values(ascending=True).tail(8)

            fig_ind_sal = go.Figure(go.Bar(
                y=ind_salary.index, x=ind_salary.values, orientation='h',
                text=[f"{v / 1000:.1f}k" for v in ind_salary.values], textposition='auto',
                marker_color='#9BBB59'
            ))
            fig_ind_sal.update_layout(
                margin=dict(l=0, r=0, t=0, b=0), height=300,
                xaxis_title="平均月薪 (元)", template="plotly_white"
            )
            st.plotly_chart(fig_ind_sal, use_container_width=True)

    # === TAB 2: 详细列表 ===
    with tab_list:
        st.markdown(f"### 📋 推荐清单详情 (共 {len(top_jobs)} 条)")

        full_columns = [
            '综合得分', '推荐理由', '职位名称', '单位名称', '薪资文本', '工作地区',
            '学历要求', '经验要求', '行业', '单位性质', '单位规模', '用工性质',
            '薪资下限', '薪资上限', '住宿情况', '发布时间', '来源类型', '职位来源', '岗位ID'
        ]
        valid_cols = [c for c in full_columns if c in top_jobs.columns]
        display_df = top_jobs[valid_cols].copy()

        if '岗位ID' in display_df.columns:
            display_df['岗位ID'] = display_df['岗位ID'].astype(str).str.replace('.0', '', regex=False)

        st.dataframe(
            display_df.style
            .format({'综合得分': "{:.1f}", '薪资下限': "{:.0f}", '薪资上限': "{:.0f}"}, na_rep="-")
            .background_gradient(subset=['综合得分'], cmap="Oranges")
            .highlight_null(color='#f0f2f6'),
            use_container_width=True,
            height=800,
            column_config={
                "岗位ID": st.column_config.TextColumn("岗位ID", help="唯一编号"),
                "薪资文本": st.column_config.TextColumn("原薪资", width="medium"),
                "单位名称": st.column_config.TextColumn("单位", width="medium"),
            }
        )

    # --- 底部操作区 ---
    st.divider()
    col_dl, col_reset = st.columns([1, 4])

    csv = top_jobs.to_csv(index=False).encode('utf-8-sig')
    col_dl.download_button(
        label="📥 下载完整数据 (Excel/CSV)",
        data=csv,
        file_name='岗位推荐结果_全字段.csv',
        mime='text/csv',
        type="primary"
    )

    if col_reset.button("🔄 重新开始测评"):
        st.session_state.step = 1
        st.rerun()
