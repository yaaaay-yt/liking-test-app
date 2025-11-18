import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import random


# 解决Matplotlib中文显示问题
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
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
        }
    </style>
""", unsafe_allow_html=True)

# --- 加载机器学习模型和标签映射 ---
try:
    model = joblib.load('emotion_model.pkl')
    label_mapping = joblib.load('label_mapping.pkl')
except FileNotFoundError:
    st.error("请先运行 `python train_model.py` 生成模型文件！")
    st.stop()

# --- 定义测试题目（与原代码保持一致） ---
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


# --- 应用主逻辑 ---
def main():
    # 初始化会话状态
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = {}
    if "test_completed" not in st.session_state:
        st.session_state.test_completed = False
    if "predicted_level" not in st.session_state:
        st.session_state.predicted_level = None

    # 侧边栏操作按钮
    with st.sidebar:
        st.markdown("### 操作面板")
        if st.button("🔄 重置测试", type="primary"):
            st.session_state.user_answers = {}
            st.session_state.test_completed = False
            st.session_state.predicted_level = None
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
                <p>你的答案将由<strong>机器学习模型</strong>分析，输出更智能的喜欢程度预测。</p>
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
                    # 准备模型输入：提取20题的得分
                    user_scores = [st.session_state.user_answers[q_id]["score"] for q_id in range(1, 21)]
                    user_input = pd.DataFrame([user_scores], columns=[f'q{i + 1}' for i in range(20)])

                    # 机器学习模型预测
                    prediction_code = model.predict(user_input)[0]
                    st.session_state.predicted_level = label_mapping[prediction_code]

                    st.session_state.test_completed = True
                    st.success("感谢你的参与！机器学习模型正在生成分析报告...")
                    st.rerun()
                else:
                    st.warning("请回答完所有问题后再提交哦~")

    # 分析结果展示逻辑
    else:
        # 结果标题
        st.markdown('<h2 class="result-title">你的喜欢程度分析报告 💝</h2>', unsafe_allow_html=True)

        # 1. 计算总分（用于进度条和参考）
        total_score = sum([ans["score"] for ans in st.session_state.user_answers.values()])
        max_score = len(questions) * 5

        # 2. 模型预测的情感等级
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

        # 3. 维度分析（与原逻辑一致）
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

        # 4. 显示总分和模型预测等级
        st.markdown(f"""
            <div class="card" style="text-align: center;">
                <h3 style="color: {color};">{emoji} {predicted_level} {emoji}</h3>
                <p style="font-size: 48px; font-weight: bold; color: {color};">
                    总分参考: {total_score}/{max_score}
                </p>
                <p style="font-size: 18px;">机器学习模型基于你的答题模式生成的预测结果</p>
            </div>
        """, unsafe_allow_html=True)

        # 5. 总分进度条
        st.markdown("### 喜欢程度可视化（总分参考）")
        st.markdown(f"""
            <div class="progress-container">
                <div class="progress-bar" style="width: {total_score / max_score * 100}%"></div>
            </div>
            <p style="text-align: center; color: #666;">
                总分进度: {total_score / max_score * 100:.1f}%
            </p>
        """, unsafe_allow_html=True)

        # 6. 维度分析雷达图
        st.markdown("### 各维度表现分析（1-5分，分数越高表示越明显）")
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

        # 准备雷达图数据
        categories = list(dimension_scores.keys())
        values = list(dimension_scores.values())
        values += values[:1]  # 闭合雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        # 绘制雷达图
        ax.plot(angles, values, 'o-', linewidth=3, color=color, markersize=8)
        ax.fill(angles, values, alpha=0.25, color=color)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)  # 这里的categories包含中文维度名称
        ax.set_ylim(0, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(["1分", "2分", "3分", "4分", "5分"], fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.title("各维度喜欢程度雷达图", fontsize=16, pad=20, fontweight='bold')

        # 添加数值标签
        for angle, value, label in zip(angles[:-1], values[:-1], categories):
            ax.text(angle, value + 0.1, str(value), ha='center', va='center', fontsize=11, fontweight='bold')

        st.pyplot(fig)

        # 7. 详细分析（与原逻辑一致，基于预测等级）
        st.markdown("### 详细分析报告")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h4>各维度详细解读：</h4>", unsafe_allow_html=True)

        for dim_name, score in dimension_scores.items():
            if score >= 4.5:
                dim_desc = "非常高"
                dim_color = "#FF4136"
                dim_emoji = "❤️"
            elif score >= 3.5:
                dim_desc = "较高"
                dim_color = "#FF851B"
                dim_emoji = "🧡"
            elif score >= 2.5:
                dim_desc = "中等"
                dim_color = "#FFDC00"
                dim_emoji = "💛"
            elif score >= 1.5:
                dim_desc = "较低"
                dim_color = "#2ECC40"
                dim_emoji = "💚"
            else:
                dim_desc = "很低"
                dim_color = "#0074D9"
                dim_emoji = "💙"

            # 维度具体解读（与原逻辑一致）
            if dim_name == "关注与记忆":
                if score >= 4:
                    dim_detail = "你会时刻关注TA的动态，记得TA的各种小细节，说明TA在你心中占据了重要位置。"
                elif score >= 3:
                    dim_detail = "你比较关注TA的动态，也能记住一些重要信息，对TA有一定的好感。"
                else:
                    dim_detail = "你对TA的关注比较有限，可能还处于初步了解阶段。"
            elif dim_name == "互动与沟通":
                if score >= 4:
                    dim_detail = "你非常愿意和TA沟通，享受和TA聊天的时光，这是喜欢一个人的重要表现。"
                elif score >= 3:
                    dim_detail = "你愿意和TA互动交流，但可能还不够主动，需要更多的勇气。"
                else:
                    dim_detail = "你和TA的互动比较少，可能还不太熟悉或者兴趣点不同。"
            elif dim_name == "情绪与感受":
                if score >= 4:
                    dim_detail = "TA对你的情绪影响很大，这说明你对TA的感情已经非常深刻了。"
                elif score >= 3:
                    dim_detail = "TA会对你的情绪产生一定影响，你对TA有明显的好感。"
                else:
                    dim_detail = "TA对你的情绪影响较小，你们之间可能更多的是朋友关系。"
            elif dim_name == "付出与支持":
                if score >= 4:
                    dim_detail = "你非常愿意为TA付出，甚至愿意调整自己的计划，这是真爱无疑了！"
                elif score >= 3:
                    dim_detail = "你愿意为TA付出一定的时间和精力，对TA有比较深的好感。"
                else:
                    dim_detail = "你在付出方面比较谨慎，可能还在观察和了解阶段。"
            else:  # 期待与想象
                if score >= 4:
                    dim_detail = "你非常期待和TA的未来，经常想象和TA在一起的场景，说明你已经深陷其中了。"
                elif score >= 3:
                    dim_detail = "你对和TA的未来有一定期待，但还比较理性，没有过度幻想。"
                else:
                    dim_detail = "你对和TA的未来没有太多期待，可能还只是把TA当作普通朋友。"

            st.markdown(f"""
                <div style="margin: 10px 0; padding: 10px; border-left: 4px solid {dim_color}; background-color: rgba(0,0,0,0.05);">
                    <h5 style="color: {dim_color};">{dim_emoji} {dim_name}：{score}分（{dim_desc}）</h5>
                    <p>{dim_detail}</p>
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 8. 情感特点分析（与原逻辑一致，基于预测等级）
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h4>情感特点分析：</h4>", unsafe_allow_html=True)

        if predicted_level == "非常喜欢":
            st.markdown("""
                <ul>
                    <li><strong>强烈的关注</strong>：你会时刻关注TA的一举一动，TA的任何动态都能引起你的注意</li>
                    <li><strong>深刻的记忆</strong>：你会记得TA说过的每一句话，甚至是一些不经意的小细节</li>
                    <li><strong>情绪的波动</strong>：TA的一句话、一个表情都能让你的情绪产生巨大波动</li>
                    <li><strong>无私的付出</strong>：你愿意为TA付出一切，甚至牺牲自己的利益</li>
                    <li><strong>未来的憧憬</strong>：你经常会想象和TA在一起的未来，对未来充满期待</li>
                </ul>
                <p style="color: #ff6b6b; margin-top: 15px;">💡 建议：你的感情已经非常明确了，可以考虑适当表白，让TA知道你的心意！</p>
            """, unsafe_allow_html=True)
        elif predicted_level == "比较喜欢":
            st.markdown("""
                <ul>
                    <li><strong>主动的关注</strong>：你会主动关注TA的动态，但还没有到时刻关注的程度</li>
                    <li><strong>良好的互动</strong>：你愿意和TA交流互动，但可能还不够自然和频繁</li>
                    <li><strong>积极的付出</strong>：你愿意为TA付出一定的时间和精力，但还会考虑自己的需求</li>
                    <li><strong>理性的期待</strong>：你对和TA的未来有期待，但还保持着一定的理性</li>
                </ul>
                <p style="color: #ff851b; margin-top: 15px;">💡 建议：可以适当增加互动频率，多了解彼此，看看感情是否能进一步发展。</p>
            """, unsafe_allow_html=True)
        elif predicted_level == "有点喜欢":
            st.markdown("""
                <ul>
                    <li><strong>偶尔的关注</strong>：你会偶尔关注TA的动态，但不会太刻意</li>
                    <li><strong>被动的互动</strong>：你愿意和TA互动，但通常是对方主动或者在特定场合</li>
                    <li><strong>谨慎的付出</strong>：你在付出方面比较谨慎，不会轻易投入太多</li>
                    <li><strong>模糊的期待</strong>：你对和TA的未来有一些期待，但还比较模糊</li>
                </ul>
                <p style="color: #ffdc00; margin-top: 15px;">💡 建议：可以多创造一些互动机会，加深了解，看看这份好感是否会变成真正的喜欢。</p>
            """, unsafe_allow_html=True)
        elif predicted_level == "好感阶段":
            st.markdown("""
                <ul>
                    <li><strong>有限的关注</strong>：你对TA的关注比较有限，可能只是在共同场合才会注意到</li>
                    <li><strong>礼貌的互动</strong>：你和TA的互动比较礼貌和表面，不会涉及太多私人话题</li>
                    <li><strong>理性的态度</strong>：你在感情方面比较理性，不会轻易投入感情</li>
                    <li><strong>开放的心态</strong>：你对和TA的关系没有太多预设，保持开放的心态</li>
                </ul>
                <p style="color: #2ecc40; margin-top: 15px;">💡 建议：保持自然的相处方式，不要有太大压力，顺其自然就好。</p>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <ul>
                    <li><strong>普通的关注</strong>：你对TA的关注和对其他朋友一样，没有特别之处</li>
                    <li><strong>朋友式互动</strong>：你和TA的互动就是普通朋友的水平，不会有太多私人交流</li>
                    <li><strong>平静的情绪</strong>：TA不会对你的情绪产生特别影响，相处起来比较平静</li>
                    <li><strong>明确的界限</strong>：你很清楚自己和TA的关系，不会有超越朋友的想法</li>
                </ul>
                <p style="color: #0074d9; margin-top: 15px;">💡 建议：保持友好的朋友关系就好，如果没有特别的感觉，不需要刻意改变什么。</p>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 9. 个性化建议（与原逻辑一致，基于预测等级）
        st.markdown("### 个性化建议")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h4>行动建议：</h4>", unsafe_allow_html=True)

        if predicted_level in ["非常喜欢", "比较喜欢"]:
            suggestions = [
                "1. <strong>主动沟通</strong>：可以更主动地和TA分享你的生活和想法，让TA更多地了解你",
                "2. <strong>创造机会</strong>：可以找一些借口创造见面机会，比如一起学习、参加活动等",
                "3. <strong>展现优点</strong>：在TA面前自然地展现你的优点和特长，让TA看到你的闪光点",
                "4. <strong>适当关心</strong>：在合适的时候给予关心和支持，让TA感受到你的温暖",
                "5. <strong>观察反应</strong>：注意观察TA对你的态度变化，如果感觉TA也有好感，可以考虑表白",
                "6. <strong>保持自我</strong>：在喜欢TA的同时，也要保持自己的独立性，不要失去自我"
            ]
        elif predicted_level == "有点喜欢":
            suggestions = [
                "1. <strong>加深了解</strong>：多了解TA的兴趣爱好，找到共同话题，增进彼此了解",
                "2. <strong>适度互动</strong>：适当增加互动频率，但不要过于频繁，给彼此一些空间",
                "3. <strong>朋友助攻</strong>：可以通过共同的朋友了解更多关于TA的信息",
                "4. <strong>耐心等待</strong>：感情需要时间培养，不要急于求成，给彼此足够的时间",
                "5. <strong>顺其自然</strong>：不要强迫自己产生特殊感觉，感情是自然而然的事情"
            ]
        elif predicted_level == "好感阶段":
            suggestions = [
                "1. <strong>自然相处</strong>：保持自然的相处方式，不要有太大压力",
                "2. <strong>共同活动</strong>：可以参加一些共同的活动，增加互动机会",
                "3. <strong>了解兴趣</strong>：多了解TA的兴趣爱好，看看是否有共同之处",
                "4. <strong>保持联系</strong>：偶尔保持联系，但不需要刻意增加互动",
                "5. <strong>关注自己</strong>：把更多精力放在自己身上，让自己变得更优秀"
            ]
        else:
            suggestions = [
                "1. <strong>保持友好</strong>：继续保持友好的朋友关系，尊重彼此的边界",
                "2. <strong>适度互动</strong>：在共同场合正常互动即可，不需要刻意增加联系",
                "3. <strong>扩大社交</strong>：可以尝试扩大自己的社交圈，认识更多的人",
                "4. <strong>关注自己</strong>：把精力放在自己的生活和成长上",
                "5. <strong>顺其自然</strong>：如果没有特别的感觉，保持现状就好"
            ]

        for suggestion in suggestions:
            st.markdown(f"<p>{suggestion}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 10. 歌曲推荐和爱情语录（与原逻辑一致）
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h4>适合你的歌曲推荐 🎵</h4>", unsafe_allow_html=True)
        songs = {
            "非常喜欢": ["《有点甜》- 汪苏泷", "《喜欢你》- Beyond", "《爱你》- 王心凌"],
            "比较喜欢": ["《小幸运》- 田馥甄", "《遇见》- 孙燕姿", "《晴天》- 周杰伦"],
            "有点喜欢": ["《有点甜》- 汪苏泷", "《暖暖》- 梁静茹", "《恋爱ing》- 五月天"],
            "好感阶段": ["《朋友》- 周华健", "《同桌的你》- 老狼", "《青春修炼手册》- TFBOYS"],
            "普通朋友": ["《友谊地久天长》", "《朋友难当》- 羽泉", "《我的好兄弟》- 高进"]
        }
        for i, song in enumerate(songs[predicted_level], 1):
            st.markdown(f"<p>{i}. {song} <span class='emoji'>🎶</span></p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        love_quotes = [
            "喜欢一个人，是看到了TA的优点；爱一个人，是包容了TA的缺点。",
            "最好的爱情，是让彼此成为更好的人。",
            "喜欢是乍见之欢，爱是久处不厌。",
            "爱情不是轰轰烈烈，而是平平淡淡中的不离不弃。",
            "真正的喜欢，是即使知道TA不完美，依然觉得TA是最好的。"
        ]
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h4>爱情语录 💌</h4>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-style: italic; color: #666;'>{random.choice(love_quotes)}</p>",
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 11. 详细回答回顾
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
                st.session_state.user_answers = {}
                st.session_state.test_completed = False
                st.session_state.predicted_level = None
                st.rerun()


if __name__ == "__main__":
    main()