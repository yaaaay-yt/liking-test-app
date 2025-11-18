import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 解决Matplotlib中文显示问题
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# --- 页面配置 ---
st.set_page_config(
    page_title="喜欢程度分析测试",
    page_icon="❤️",
    layout="centered"
)

# --- 自定义CSS：动态背景、卡片样式等 ---
st.markdown("""
    <style>
        .animated-bg {
            background: linear-gradient(45deg, #ff9a9e, #fad0c4, #fad0c4, #ffd1ff);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            opacity: 0.3;
        }
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .card {
            background-color: rgba(255, 255, 255, 0.85);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        .card:hover { transform: translateY(-5px); }
        .emoji {
            font-size: 24px;
            animation: bounce 2s infinite;
            display: inline-block;
            margin: 0 5px;
        }
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        .progress-container {
            background-color: #e0e0e0;
            border-radius: 20px;
            height: 20px;
            width: 100%;
            margin: 10px 0;
            overflow: hidden;
        }
        .progress-bar {
            background: linear-gradient(90deg, #ff9a9e, #f6416c);
            height: 100%;
            border-radius: 20px;
            transition: width 1s ease-in-out;
        }
        .result-title {
            font-size: 28px;
            font-weight: bold;
            text-align: center;
            margin: 20px 0;
            background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            animation: fadeInGlow 2s ease-in-out;
        }
        @keyframes fadeInGlow {
            from { opacity: 0; text-shadow: 0 0 20px transparent; }
            to { opacity: 1; text-shadow: 0 0 5px rgba(255, 107, 107, 0.5), 0 0 10px rgba(78, 205, 196, 0.5); }
        }
        .analysis-card {
            animation: slideUp 0.8s ease-out;
        }
        @keyframes slideUp {
            from { transform: translateY(30px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        .song-item {
            display: flex;
            align-items: center;
            opacity: 0;
            animation: fadeIn 0.5s forwards;
        }
        .song-item:nth-child(1) { animation-delay: 0.1s; }
        .song-item:nth-child(2) { animation-delay: 0.2s; }
        .song-item:nth-child(3) { animation-delay: 0.3s; }
        .song-item:nth-child(4) { animation-delay: 0.4s; }
        .song-item:nth-child(5) { animation-delay: 0.5s; }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateX(-10px); }
            to { opacity: 1; transform: translateX(0); }
        }
        .emoji-spin {
            display: inline-block;
            animation: spin 10s linear infinite;
        }
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
    </style>
""", unsafe_allow_html=True)

# --- 定义测试题目 ---
questions = [
    {"id": 1, "text": "我会不自觉地关注TA的社交媒体动态（如朋友圈、微博）",
     "options": [("完全不会", 1), ("偶尔会", 2), ("有时会", 3), ("经常会", 4), ("总是会", 5)]},
    {"id": 2, "text": "和TA聊天时，我会感到很放松，愿意分享很多事情",
     "options": [("完全不会", 1), ("偶尔会", 2), ("有时会", 3), ("经常会", 4), ("总是会", 5)]},
    {"id": 3, "text": "看到TA和别人互动频繁时，我会有一点点不自在",
     "options": [("完全不会", 1), ("偶尔会", 2), ("有时会", 3), ("经常会", 4), ("总是会", 5)]},
    {"id": 4, "text": "我会记得TA提过的一些小喜好（比如喜欢的食物、电影）",
     "options": [("完全不会", 1), ("偶尔会", 2), ("有时会", 3), ("经常会", 4), ("总是会", 5)]},
    {"id": 5, "text": "我期待和TA有更多的见面或交流机会",
     "options": [("完全不期待", 1), ("偶尔期待", 2), ("有时期待", 3), ("经常期待", 4), ("非常期待", 5)]},
    {"id": 6, "text": "听到TA的名字时，我的情绪会有明显波动（开心、紧张等）",
     "options": [("完全不会", 1), ("偶尔会", 2), ("有时会", 3), ("经常会", 4), ("总是会", 5)]},
    {"id": 7, "text": "我会主动找话题和TA聊天，哪怕只是简单的问候",
     "options": [("完全不会", 1), ("偶尔会", 2), ("有时会", 3), ("经常会", 4), ("总是会", 5)]},
    {"id": 8, "text": "我会在生活中留意适合TA的小礼物或小惊喜", "options": [("完全不会", 1), ("偶尔会", 2), ("有时会", 3), ("经常会", 4), ("总是会", 5)]},
    {"id": 9, "text": "当TA遇到困难时，我会很想帮TA解决", "options": [("完全不想", 1), ("偶尔想", 2), ("有时想", 3), ("经常想", 4), ("非常想", 5)]},
    {"id": 10, "text": "我会想象和TA一起做某件事的场景（比如一起旅行、看电影）",
     "options": [("完全不会", 1), ("偶尔会", 2), ("有时会", 3), ("经常会", 4), ("总是会", 5)]},
    {"id": 11, "text": "我会因为TA的一句夸奖而开心很久", "options": [("完全不会", 1), ("偶尔会", 2), ("有时会", 3), ("经常会", 4), ("总是会", 5)]},
    {"id": 12, "text": "我会下意识地模仿TA的一些习惯或说话方式",
     "options": [("完全不会", 1), ("偶尔会", 2), ("有时会", 3), ("经常会", 4), ("总是会", 5)]},
    {"id": 13, "text": "我会关注TA的情绪变化，想知道TA为什么开心或难过",
     "options": [("完全不会", 1), ("偶尔会", 2), ("有时会", 3), ("经常会", 4), ("总是会", 5)]},
    {"id": 14, "text": "我愿意为了和TA有共同话题，去了解TA感兴趣的领域",
     "options": [("完全不愿意", 1), ("偶尔愿意", 2), ("有时愿意", 3), ("经常愿意", 4), ("非常愿意", 5)]},
    {"id": 15, "text": "当TA出现在视线里时，我的注意力会不自觉地被吸引",
     "options": [("完全不会", 1), ("偶尔会", 2), ("有时会", 3), ("经常会", 4), ("总是会", 5)]},
    {"id": 16, "text": "我会在朋友面前不自觉地提起TA", "options": [("完全不会", 1), ("偶尔会", 2), ("有时会", 3), ("经常会", 4), ("总是会", 5)]},
    {"id": 17, "text": "我会因为TA的存在，觉得某个场合变得更有趣",
     "options": [("完全不会", 1), ("偶尔会", 2), ("有时会", 3), ("经常会", 4), ("总是会", 5)]},
    {"id": 18, "text": "我会在深夜里回忆和TA的对话片段", "options": [("完全不会", 1), ("偶尔会", 2), ("有时会", 3), ("经常会", 4), ("总是会", 5)]},
    {"id": 19, "text": "我觉得TA身上有很多值得我欣赏的优点",
     "options": [("完全不觉得", 1), ("偶尔觉得", 2), ("有时觉得", 3), ("经常觉得", 4), ("非常觉得", 5)]},
    {"id": 20, "text": "如果TA需要帮助，我会愿意调整自己的计划去支持TA",
     "options": [("完全不愿意", 1), ("偶尔愿意", 2), ("有时愿意", 3), ("经常愿意", 4), ("非常愿意", 5)]}
]


# --- 核心修改：基于总分判断情感等级的函数 ---
def determine_level(total_score):
    if total_score >= 90:
        return "非常喜欢"
    elif total_score >= 75:
        return "比较喜欢"
    elif total_score >= 60:
        return "有点喜欢"
    elif total_score >= 40:
        return "好感阶段"
    else:
        return "普通朋友"


# --- 应用主逻辑 ---
def main():
    # 初始化会话状态
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = {}
    if "test_completed" not in st.session_state:
        st.session_state.test_completed = False
    if "predicted_level" not in st.session_state:
        st.session_state.predicted_level = None
    if "total_score" not in st.session_state:
        st.session_state.total_score = 0

    # 侧边栏操作按钮
    with st.sidebar:
        st.markdown("### 操作面板")
        if st.button("🔄 重置测试", type="primary"):
            for key in ["user_answers", "test_completed", "predicted_level", "total_score"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    # 添加动态背景
    st.markdown('<div class="animated-bg"></div>', unsafe_allow_html=True)

    # 测试题展示逻辑
    if not st.session_state.test_completed:
        # 标题与动画表情
        st.markdown("""
            <div style="text-align: center; margin-bottom: 30px;">
                <h1 style="font-size: 40px; background: linear-gradient(90deg, #ff6b6b, #4ecdc4); -webkit-background-clip: text; background-clip: text; color: transparent;">
                    ❤️ 对某个人的喜欢程度分析测试 ❤️
                </h1>
                <p style="font-size: 20px;">
                    <span class="emoji">💘</span>
                    <span class="emoji">💝</span>
                    <span class="emoji">💖</span>
                    <span class="emoji">💗</span>
                    <span class="emoji">💓</span>
                </p>
            </div>
        """, unsafe_allow_html=True)

        # 测试说明卡片
        st.markdown("""
            <div class="card">
                <h3 style="color: #ff6b6b;">测试说明 💌</h3>
                <p>请阅读以下20个陈述，并根据你的真实感受选择最符合的选项。</p>
                <p>系统将根据你的总分，为你分析喜欢程度。</p>
                <p style="color: #4ecdc4; font-style: italic;">请诚实作答，结果将更准确地反映你的真实感受哦~</p>
            </div>
        """, unsafe_allow_html=True)

        # 进度条
        progress = len(st.session_state.user_answers) / len(questions)
        st.markdown(f"""
            <div class="progress-container">
                <div class="progress-bar" style="width: {progress * 100}%"></div>
            </div>
            <p style="text-align: center; color: #666;">
                当前进度: {len(st.session_state.user_answers)}/{len(questions)} 题
            </p>
        """, unsafe_allow_html=True)

        # 逐个渲染题目
        for question in questions:
            q_id = question["id"]
            q_text = question["text"]
            q_options = question["options"]

            st.markdown(f"""
                <div class="card">
                    <h4 style="color: #ff6b6b;">问题 {q_id}: {q_text}</h4>
                </div>
            """, unsafe_allow_html=True)

            # 加载已有答案或默认选第一个
            default_index = 0
            if q_id in st.session_state.user_answers:
                user_answer_text = st.session_state.user_answers[q_id]["text"]
                for i, (option_text, _) in enumerate(q_options):
                    if option_text == user_answer_text:
                        default_index = i
                        break

            selected_option_text = st.radio(
                "请选择你的态度：",
                [option[0] for option in q_options],
                key=f"question_{q_id}",
                index=default_index
            )

            # 存储答案的文本和分数
            for option_text, option_score in q_options:
                if selected_option_text == option_text:
                    st.session_state.user_answers[q_id] = {
                        "text": option_text,
                        "score": option_score
                    }
                    break

        # 提交按钮
        col_submit = st.columns([1, 2, 1])[1]
        with col_submit:
            if st.button("提交我的答案 💖", type="primary"):
                if len(st.session_state.user_answers) == len(questions):
                    # 计算总分
                    total_score = sum([ans["score"] for ans in st.session_state.user_answers.values()])
                    st.session_state.total_score = total_score

                    # --- 核心修改：使用规则函数判断等级 ---
                    st.session_state.predicted_level = determine_level(total_score)

                    st.session_state.test_completed = True
                    st.success("感谢你的参与！正在生成分析报告...")
                    st.rerun()
                else:
                    st.warning("请回答完所有问题后再提交哦~")

    # 分析结果展示逻辑
    else:
        # 结果标题
        st.markdown('<h2 class="result-title">你的喜欢程度分析报告 💝</h2>', unsafe_allow_html=True)

        # 1. 显示总分
        total_score = st.session_state.total_score
        max_score = len(questions) * 5

        # 2. 显示情感等级
        predicted_level = st.session_state.predicted_level
        level_info = {
            "非常喜欢": {"color": "#FF4136", "emoji": "❤️"},
            "比较喜欢": {"color": "#FF851B", "emoji": "🧡"},
            "有点喜欢": {"color": "#FFDC00", "emoji": "💛"},
            "好感阶段": {"color": "#2ECC40", "emoji": "💚"},
            "普通朋友": {"color": "#0074D9", "emoji": "💙"}
        }
        color = level_info[predicted_level]["color"]
        emoji = level_info[predicted_level]["emoji"]

        # 3. 维度分析（逻辑保持不变）
        dimensions = {
            "关注与记忆": [1, 4, 13, 15, 18],
            "互动与沟通": [2, 7, 12, 16, 17],
            "情绪与感受": [3, 6, 10, 11, 19],
            "付出与支持": [8, 9, 14, 20],
            "期待与想象": [5]
        }
        dimension_scores = {}
        for dim_name, q_ids in dimensions.items():
            dim_score = sum([st.session_state.user_answers[q_id]["score"] for q_id in q_ids])
            dim_avg = dim_score / len(q_ids)
            dimension_scores[dim_name] = round(dim_avg, 1)

        # 4. 显示总分和等级
        st.markdown(f"""
            <div class="card" style="text-align: center;">
                <h3 style="color: {color};">{emoji} {predicted_level} {emoji}</h3>
                <p style="font-size: 48px; font-weight: bold; color: {color};">
                    总分: {total_score}/{max_score}
                </p>
                <p style="font-size: 18px;">根据你的答题情况分析得出</p>
            </div>
        """, unsafe_allow_html=True)

        # 5. 总分进度条
        st.markdown("### 喜欢程度可视化")
        st.markdown(f"""
            <div class="progress-container">
                <div class="progress-bar" style="width: {total_score / max_score * 100}%"></div>
            </div>
            <p style="text-align: center; color: #666;">
                匹配度: {total_score / max_score * 100:.1f}%
            </p>
        """, unsafe_allow_html=True)

        # 6. 维度分析雷达图
        st.markdown("### 各维度表现分析（1-5分，分数越高表示越明显）")
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        categories = list(dimension_scores.keys())
        values = list(dimension_scores.values())
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        ax.plot(angles, values, 'o-', linewidth=3, color=color, markersize=8)
        ax.fill(angles, values, alpha=0.25, color=color)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylim(0, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(["1分", "2分", "3分", "4分", "5分"], fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.title("各维度喜欢程度雷达图", fontsize=16, pad=20, fontweight='bold')
        for angle, value, label in zip(angles[:-1], values[:-1], categories):
            ax.text(angle, value + 0.1, str(value), ha='center', va='center', fontsize=11, fontweight='bold')
        st.pyplot(fig)

        # 7. 详细分析
        st.markdown("### 详细分析报告")
        st.markdown('<div class="card analysis-card">', unsafe_allow_html=True)
        st.markdown("<h4>各维度详细解读：</h4>", unsafe_allow_html=True)
        for dim_name, score in dimension_scores.items():
            if score >= 4.5:
                dim_desc, dim_color, dim_emoji = "非常高", "#FF4136", "❤️"
            elif score >= 3.5:
                dim_desc, dim_color, dim_emoji = "较高", "#FF851B", "🧡"
            elif score >= 2.5:
                dim_desc, dim_color, dim_emoji = "中等", "#FFDC00", "💛"
            elif score >= 1.5:
                dim_desc, dim_color, dim_emoji = "较低", "#2ECC40", "💚"
            else:
                dim_desc, dim_color, dim_emoji = "很低", "#0074D9", "💙"

            dim_detail = ""
            if dim_name == "关注与记忆":
                dim_detail = "你会时刻关注TA的动态，记得TA的各种小细节。" if score >= 4 else "你比较关注TA的动态，也能记住一些重要信息。" if score >= 3 else "你对TA的关注比较有限。"
            elif dim_name == "互动与沟通":
                dim_detail = "你非常愿意和TA沟通，享受和TA聊天的时光。" if score >= 4 else "你愿意和TA互动交流，但可能还不够主动。" if score >= 3 else "你和TA的互动比较少。"
            elif dim_name == "情绪与感受":
                dim_detail = "TA对你的情绪影响很大，这说明你对TA的感情已经非常深刻了。" if score >= 4 else "TA会对你的情绪产生一定影响，你对TA有明显的好感。" if score >= 3 else "TA对你的情绪影响较小。"
            elif dim_name == "付出与支持":
                dim_detail = "你非常愿意为TA付出，甚至愿意调整自己的计划。" if score >= 4 else "你愿意为TA付出一定的时间和精力。" if score >= 3 else "你在付出方面比较谨慎。"
            else:  # 期待与想象
                dim_detail = "你非常期待和TA的未来，经常想象和TA在一起的场景。" if score >= 4 else "你对和TA的未来有一定期待，但还比较理性。" if score >= 3 else "你对和TA的未来没有太多期待。"

            st.markdown(f"""
                <div style="margin: 10px 0; padding: 10px; border-left: 4px solid {dim_color}; background-color: rgba(0,0,0,0.05);">
                    <h5 style="color: {dim_color};">{dim_emoji} {dim_name}：{score}分（{dim_desc}）</h5>
                    <p>{dim_detail}</p>
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 8. 情感特点分析
        st.markdown("### 情感特点分析")
        st.markdown('<div class="card analysis-card">', unsafe_allow_html=True)
        if predicted_level == "非常喜欢":
            st.markdown(f"""
                <p>你的情感已经非常明确，TA在你心中占据了重要的位置。你不仅在行为上表现出强烈的关注和付出意愿，
                而且在情绪上也深受TA的影响。这种喜欢是全面且深刻的，值得你认真考虑如何进一步发展关系。</p>
                <p style="text-align: center; font-size: 24px;">
                    <span class="emoji-spin">❤️</span>
                    <span class="emoji-spin" style="animation-delay: 1s;">💖</span>
                    <span class="emoji-spin" style="animation-delay: 2s;">💘</span>
                </p>
            """, unsafe_allow_html=True)
        elif predicted_level == "比较喜欢":
            st.markdown("""
                <p>你对TA抱有明显的好感，并且这种感觉在多个方面都有所体现。你会主动关注TA、愿意与TA互动，
                也会为TA付出一定的精力。你正处于一个对TA逐渐加深了解和喜欢的阶段。</p>
                <p style="text-align: center; font-size: 24px;">
                    <span class="emoji">💞</span>
                    <span class="emoji">💝</span>
                    <span class="emoji">💟</span>
                </p>
            """, unsafe_allow_html=True)
        elif predicted_level == "有点喜欢":
            st.markdown("""
                <p>你对TA产生了初步的好感，在某些方面会特别关注TA。这种喜欢还比较含蓄，可能还处于试探和观察的阶段。
                随着更多的了解和互动，这种感觉可能会进一步加深。</p>
                <p style="text-align: center; font-size: 24px;">
                    <span class="emoji">💓</span>
                    <span class="emoji">💗</span>
                    <span class="emoji">💔</span>
                </p>
            """, unsafe_allow_html=True)
        elif predicted_level == "好感阶段":
            st.markdown("""
                <p>你对TA有一定的好感，但这种感觉还比较轻微和表面。你可能会注意到TA的存在，
                但在主动互动和情感投入上还比较克制。这是一个很好的起点，可以通过更多的接触来增进了解。</p>
                <p style="text-align: center; font-size: 24px;">
                    <span class="emoji">💚</span>
                    <span class="emoji">💙</span>
                    <span class="emoji">💛</span>
                </p>
            """, unsafe_allow_html=True)
        else:  # 普通朋友
            st.markdown("""
                <p>目前来看，你对TA更多的是朋友般的感觉。你会以一种轻松、自然的方式与TA相处，
                没有表现出特别强烈的情感倾向。这种关系状态也很美好，可以保持轻松的互动，顺其自然发展。</p>
                <p style="text-align: center; font-size: 24px;">
                    <span class="emoji">🤝</span>
                    <span class="emoji">👫</span>
                    <span class="emoji">👬</span>
                </p>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 9. 个性化建议（行动建议）
        st.markdown("### 行动建议")
        st.markdown('<div class="card analysis-card">', unsafe_allow_html=True)
        if predicted_level == "非常喜欢":
            st.markdown("""
                <ol>
                    <li>勇敢表达：找合适的时机，真诚地表达你的心意，不要让缘分错过。</li>
                    <li>深度互动：创造更多独处或深度交流的机会，了解TA的内心世界。</li>
                    <li>展现自我：在互动中展现你的优点和魅力，让TA看到你的闪光点。</li>
                    <li>尊重节奏：如果TA暂时没有回应，尊重TA的节奏，给彼此时间。</li>
                    <li>制造回忆：一起做一些有意义的事情，创造共同的美好回忆。</li>
                </ol>
            """, unsafe_allow_html=True)
        elif predicted_level == "比较喜欢":
            st.markdown("""
                <ol>
                    <li>升温关系：从日常关心开始，逐步增加互动的频率和深度，让关系自然升温。</li>
                    <li>兴趣共鸣：找到你们的共同兴趣点，以此为桥梁增进彼此的了解和好感。</li>
                    <li>适度试探：可以通过一些小举动试探TA的态度，比如分享心事、制造小惊喜。</li>
                    <li>自我提升：在关注TA的同时，也别忘了提升自己，让自己更具吸引力。</li>
                    <li>保持真诚：无论关系如何发展，都保持真诚的态度，这是感情的基础。</li>
                </ol>
            """, unsafe_allow_html=True)
        elif predicted_level == "有点喜欢":
            st.markdown("""
                <ol>
                    <li>增加接触：创造更多“偶遇”或集体活动的机会，自然地增加与TA的接触。</li>
                    <li>轻松聊天：从轻松的话题入手，比如兴趣爱好、生活趣事，慢慢拉近距离。</li>
                    <li>默默关注：在TA需要帮助时，及时伸出援手，让TA感受到你的在意。</li>
                    <li>观察细节：留意TA的喜好和习惯，为后续的互动找到突破口。</li>
                    <li>放平心态：不要急于求成，享受这个慢慢了解和心动的过程。</li>
                </ol>
            """, unsafe_allow_html=True)
        elif predicted_level == "好感阶段":
            st.markdown("""
                <ol>
                    <li>自然互动：在日常相处中保持自然、友好的互动，不要过于刻意。</li>
                    <li>拓展话题：除了日常寒暄，尝试聊一些更深入的话题，了解彼此的价值观。</li>
                    <li>共同活动：一起参加一些轻松的集体活动，比如看电影、爬山，在活动中增进感情。</li>
                    <li>保持距离：在建立好感的初期，适当的距离也很重要，给彼此空间。</li>
                    <li>耐心观察：多观察TA的为人和对待他人的态度，再决定是否进一步发展。</li>
                </ol>
            """, unsafe_allow_html=True)
        else:  # 普通朋友
            st.markdown("""
                <ol>
                    <li>保持友好：继续保持友好的朋友关系，尊重彼此的边界。</li>
                    <li>适度互动：在共同场合正常互动即可，不需要刻意增加联系。</li>
                    <li>扩大社交：可以尝试扩大自己的社交圈，认识更多的人。</li>
                    <li>关注自己：把精力放在自己的生活和成长上。</li>
                    <li>顺其自然：如果没有特别的感觉，保持现状就好。</li>
                </ol>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 10. 歌曲推荐
        st.markdown("### 适合你的歌曲推荐 🎵")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if predicted_level == "非常喜欢":
            songs = [
                "《有点甜》- 汪苏泷/By2",
                "《小幸运》- 田馥甄",
                "《爱你》- 王心凌",
                "《告白气球》- 周杰伦",
                "《说爱你》- 蔡依林"
            ]
        elif predicted_level == "比较喜欢":
            songs = [
                "《遇见》- 孙燕姿",
                "《晴天》- 周杰伦",
                "《温柔》- 五月天",
                "《小酒窝》- 林俊杰/蔡卓妍",
                "《不得不爱》- 潘玮柏/弦子"
            ]
        elif predicted_level == "有点喜欢":
            songs = [
                "《七里香》- 周杰伦",
                "《后来》- 刘若英",
                "《稻香》- 周杰伦",
                "《青春修炼手册》- TFBOYS",
                "《宠爱》- TFBOYS"
            ]
        elif predicted_level == "好感阶段":
            songs = [
                "《朋友》- 周华健",
                "《同桌的你》- 老狼",
                "《最初的梦想》- 范玮琪",
                "《隐形的翅膀》- 张韶涵",
                "《阳光彩虹小白马》- 大张伟"
            ]
        else:  # 普通朋友
            songs = [
                "《友谊地久天长》- 群星",
                "《朋友难当》- 羽泉",
                "《我的好兄弟》- 高进",
                "《我的未来不是梦》- 张雨生",
                "《海阔天空》- Beyond"
            ]

        for i, song in enumerate(songs, 1):
            st.markdown(
                f'<div class="song-item"><span style="margin-right: 10px; color: {color}; font-weight: bold;">{i}. </span>{song}</div>',
                unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # 11. 爱情语录
        st.markdown("### 爱情语录 💕")
        st.markdown('<div class="card analysis-card">', unsafe_allow_html=True)
        if predicted_level == "非常喜欢":
            st.markdown("最好的爱情，是让彼此成为更好的人。")
        elif predicted_level == "比较喜欢":
            st.markdown("喜欢是棋逢对手，爱是甘拜下风。")
        elif predicted_level == "有点喜欢":
            st.markdown("喜欢是一朝一夕，爱是从心动到古稀。")
        elif predicted_level == "好感阶段":
            st.markdown("相遇是缘，相处是份，缘分需要珍惜。")
        else:  # 普通朋友
            st.markdown("最好的友情，是各自忙碌，互相牵挂。")
        st.markdown('</div>', unsafe_allow_html=True)

        # 12. 详细回答回顾
        st.markdown("### 你的详细回答")
        answer_data = []
        for question in questions:
            q_id = question["id"]
            q_text = question["text"]
            user_answer = st.session_state.user_answers[q_id]["text"]
            answer_data.append({"问题": q_text, "你的回答": user_answer})
        answer_df = pd.DataFrame(answer_data)
        st.dataframe(answer_df, use_container_width=True)

        # 重新测试按钮
        col_restart = st.columns([1, 2, 1])[1]
        with col_restart:
            if st.button("🔄 返回重新测试", type="primary"):
                for key in ["user_answers", "test_completed", "predicted_level", "total_score"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()


if __name__ == "__main__":

    main()


