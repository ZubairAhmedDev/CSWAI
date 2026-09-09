import time

import ollama
import pandas as pd
import streamlit as st

from core.db import (
    init_db,
    ensure_demo_students,
    seed_demo_performance,
    log_interaction,
    save_assessment,
    get_connection,
)
from core.topics import detect_topic
from core.rag import (
    build_index,
    INDEX_PATH,
    knowledge_base_stats,
    retrieve_with_diagnostics,
)
from core.llm import answer_with_context, CHAT_MODEL
from core.analytics import (
    student_overview,
    recommendation,
    class_overview,
)
from core.security import (
    inspect_user_query,
    sanitize_retrieved_chunks,
    SessionRateLimiter,
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CSWAI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# GLASSMORPHISM UI
# =========================================================

st.markdown(
    """
<style>
html, body, [class*="css"] {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                 BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 15% 10%,
            rgba(91, 80, 255, 0.16), transparent 28%),
        radial-gradient(circle at 85% 15%,
            rgba(0, 209, 255, 0.10), transparent 25%),
        radial-gradient(circle at 70% 85%,
            rgba(124, 58, 237, 0.10), transparent 30%),
        linear-gradient(
            145deg,
            #080b12 0%,
            #0c111b 42%,
            #101522 100%
        );
    background-attachment: fixed;
}

.block-container {
    max-width: 1350px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: rgba(18, 22, 34, 0.60);
    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] > div:first-child {
    background: transparent;
}

.cswai-hero {
    padding: 1.5rem 1.7rem;
    margin-bottom: 1.25rem;
    border-radius: 22px;
    background: linear-gradient(
        135deg,
        rgba(255,255,255,0.10),
        rgba(255,255,255,0.045)
    );
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow:
        0 18px 45px rgba(0,0,0,0.28),
        inset 0 1px 0 rgba(255,255,255,0.06);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
}

.cswai-title {
    font-size: 2.65rem;
    line-height: 1;
    font-weight: 800;
    letter-spacing: -0.04em;
    margin: 0;
}

.cswai-subtitle {
    margin-top: 0.65rem;
    color: rgba(235,240,255,0.68);
    font-size: 1rem;
}

.cswai-badge {
    display: inline-block;
    margin-top: 0.9rem;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    color: rgba(245,248,255,0.84);
    font-size: 0.78rem;
}

[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        rgba(255,255,255,0.095),
        rgba(255,255,255,0.04)
    );
    border: 1px solid rgba(255,255,255,0.11);
    border-radius: 18px;
    padding: 16px 18px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.20);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
}

.stButton > button {
    border-radius: 13px;
    border: 1px solid rgba(255,255,255,0.12);
    background: linear-gradient(
        135deg,
        rgba(255,255,255,0.095),
        rgba(255,255,255,0.045)
    );
    box-shadow: 0 8px 22px rgba(0,0,0,0.18);
}

[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 17px;
    backdrop-filter: blur(16px);
}

[data-testid="stChatMessage"] {
    background: linear-gradient(
        145deg,
        rgba(255,255,255,0.075),
        rgba(255,255,255,0.032)
    );
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 0.45rem 0.85rem;
    margin-bottom: 0.65rem;
    box-shadow: 0 10px 28px rgba(0,0,0,0.16);
    backdrop-filter: blur(14px);
}

[data-testid="stExpander"] {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
}

div[data-testid="stAlert"] {
    border-radius: 14px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.07);
    padding: 0.35rem;
    border-radius: 14px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
}

[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 15px;
    overflow: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================

def hero(title, subtitle, badge):
    st.markdown(
        f"""
        <div class="cswai-hero">
            <div class="cswai-title">{title}</div>
            <div class="cswai-subtitle">{subtitle}</div>
            <div class="cswai-badge">{badge}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ollama_status():
    try:
        result = ollama.list()

        models = getattr(
            result,
            "models",
            None,
        )

        if (
            models is None
            and isinstance(result, dict)
        ):
            models = result.get(
                "models",
                [],
            )

        names = []

        for item in models or []:
            name = getattr(
                item,
                "model",
                None,
            )

            if (
                not name
                and isinstance(item, dict)
            ):
                name = (
                    item.get("model")
                    or item.get("name")
                )

            if name:
                names.append(name)

        return True, names

    except Exception:
        return False, []


def render_sources(chunks):
    if not chunks:
        return

    with st.expander(
        "Supporting evidence",
        expanded=False,
    ):
        for index, item in enumerate(
            chunks,
            start=1,
        ):
            st.markdown(
                f"**{index}. {item['source']} - "
                f"page {item['page']}**  \n"
                f"`{item.get('topic', 'General')}` | "
                f"`{item.get('subtopic', 'General')}` | "
                f"`{item.get('eke_layer', 'CONCEPT')}`"
            )

            excerpt = item.get(
                "text",
                "",
            )

            if len(excerpt) > 700:
                excerpt = (
                    excerpt[:700].rstrip()
                    + "..."
                )

            st.caption(excerpt)

            score_parts = []

            if "semantic_score" in item:
                score_parts.append(
                    f"semantic "
                    f"{item['semantic_score']:.3f}"
                )

            if "bm25_score" in item:
                score_parts.append(
                    f"BM25 "
                    f"{item['bm25_score']:.3f}"
                )

            if "score" in item:
                score_parts.append(
                    f"hybrid "
                    f"{item['score']:.3f}"
                )

            if score_parts:
                st.caption(
                    " | ".join(
                        score_parts
                    )
                )

            if index < len(chunks):
                st.divider()


# =========================================================
# INITIALIZATION
# =========================================================

init_db()
ensure_demo_students()
seed_demo_performance()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "rate_limiter" not in st.session_state:
    st.session_state.rate_limiter = (
        SessionRateLimiter()
    )

online, _ = ollama_status()
kb = knowledge_base_stats()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("CSWAI")
st.sidebar.caption(
    "Engineering Education AI"
)

portal = st.sidebar.radio(
    "Portal",
    [
        "Student Portal",
        "Instructor Portal",
    ],
)

student_id = st.sidebar.selectbox(
    "Demo student",
    [
        f"S{i:03d}"
        for i in range(1, 31)
    ],
)

st.sidebar.divider()

if online:
    st.sidebar.success(
        "Local AI online"
    )
else:
    st.sidebar.error(
        "Local AI offline"
    )

if INDEX_PATH.exists():
    st.sidebar.success(
        f"Knowledge base ready | "
        f"{kb['chunks']} chunks"
    )
else:
    st.sidebar.warning(
        "Knowledge base not indexed"
    )

st.sidebar.caption(
    f"Tutor model | {CHAT_MODEL}"
)


# =========================================================
# STUDENT PORTAL
# =========================================================

if portal == "Student Portal":
    hero(
        "CSWAI",
        (
            "Offline AI Platform for SOLIDWORKS "
            "and CSWA Education"
        ),
        (
            "Student Portal | Grounded Engineering Tutor | "
            "Injection Guard Enabled"
        ),
    )

    page = st.radio(
        "Student tools",
        [
            "Ask AI",
            "My Progress",
            "Practice Quiz",
            "Certification Progress",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    # -----------------------------------------------------
    # ASK AI
    # -----------------------------------------------------

    if page == "Ask AI":
        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Knowledge Documents",
            kb["documents"],
        )

        m2.metric(
            "Indexed Pages",
            kb["pages"],
        )

        m3.metric(
            "Knowledge Chunks",
            kb["chunks"],
        )

        st.subheader(
            "Engineering AI Tutor"
        )

        st.caption(
            "Grounded in approved CSWAI knowledge. "
            "Prompt-injection screening is enabled."
        )

        suggestions = [
            "What is an Extruded Boss/Base?",
            "How do I fully define a sketch?",
            "What is a concentric mate?",
            "What is a section view?",
        ]

        columns = st.columns(4)

        for column, question in zip(
            columns,
            suggestions,
        ):
            if column.button(
                question,
                use_container_width=True,
            ):
                st.session_state.pending_question = (
                    question
                )

        if st.button(
            "Clear conversation"
        ):
            st.session_state.messages = []
            st.rerun()

        for message in st.session_state.messages:
            with st.chat_message(
                message["role"]
            ):
                st.markdown(
                    message["content"]
                )

                if (
                    message["role"]
                    == "assistant"
                ):
                    meta = message.get(
                        "meta",
                        {},
                    )

                    if meta:
                        st.caption(
                            f"Evidence confidence | "
                            f"{meta.get('confidence', '-')}   |   "
                            f"Topic | "
                            f"{meta.get('topic', 'General')}   |   "
                            f"Response | "
                            f"{meta.get('latency', 0):.1f}s"
                        )

                    render_sources(
                        message.get(
                            "sources",
                            [],
                        )
                    )

        typed_question = st.chat_input(
            "Ask a SOLIDWORKS or CSWA question"
        )

        prompt = (
            st.session_state.pending_question
            or typed_question
        )

        if prompt:
            st.session_state.pending_question = None

            if not st.session_state.rate_limiter.allow():
                st.warning(
                    "Too many requests in a short period. "
                    "Please wait about a minute and try again."
                )
                st.stop()

            security = inspect_user_query(
                prompt
            )

            if not security.allowed:
                if security.reason.startswith(
                    "query_too_long"
                ):
                    st.warning(
                        "Your question is too long. "
                        "Please shorten it and try again."
                    )
                else:
                    st.warning(
                        "CSWAI blocked this request because it "
                        "appears to contain instructions that "
                        "attempt to override or expose the tutor's "
                        "security controls."
                    )

                st.stop()

            prompt = (
                security.cleaned_text
            )

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )

            with st.chat_message(
                "user"
            ):
                st.markdown(
                    prompt
                )

            started = (
                time.perf_counter()
            )

            topic = detect_topic(
                prompt
            )

            with st.chat_message(
                "assistant"
            ):
                with st.spinner(
                    "Searching approved engineering knowledge..."
                ):
                    try:
                        diagnostic = (
                            retrieve_with_diagnostics(
                                prompt,
                                top_k=5,
                            )
                        )

                        chunks = (
                            diagnostic[
                                "results"
                            ]
                        )

                        (
                            chunks,
                            blocked_chunks,
                        ) = sanitize_retrieved_chunks(
                            chunks
                        )

                        if not chunks:
                            answer = (
                                "I could not find sufficient evidence "
                                "in the approved CSWAI knowledge base "
                                "to answer that question reliably."
                            )

                        else:
                            answer = (
                                answer_with_context(
                                    prompt,
                                    chunks,
                                    confidence=diagnostic.get(
                                        "confidence",
                                        "Medium",
                                    ),
                                )
                            )

                        latency = (
                            time.perf_counter()
                            - started
                        )

                    except Exception as exc:
                        diagnostic = {
                            "confidence": "Unavailable",
                            "query_topic": topic,
                        }

                        chunks = []

                        latency = (
                            time.perf_counter()
                            - started
                        )

                        answer = (
                            "The local CSWAI tutor encountered "
                            "an error while processing this request: "
                            f"{exc}"
                        )

                st.markdown(
                    answer
                )

                st.caption(
                    f"Evidence confidence | "
                    f"{diagnostic.get('confidence', '-')}   |   "
                    f"Topic | "
                    f"{diagnostic.get('query_topic', topic)}   |   "
                    f"Response | "
                    f"{latency:.1f}s"
                )

                render_sources(
                    chunks
                )

            log_interaction(
                student_id,
                prompt,
                topic,
                answer,
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": chunks,
                    "meta": {
                        "confidence": (
                            diagnostic.get(
                                "confidence",
                                "-",
                            )
                        ),
                        "topic": (
                            diagnostic.get(
                                "query_topic",
                                topic,
                            )
                        ),
                        "latency": latency,
                    },
                }
            )

    # -----------------------------------------------------
    # MY PROGRESS
    # -----------------------------------------------------

    elif page == "My Progress":
        overview = student_overview(
            student_id
        )

        rec = recommendation(
            student_id
        )

        st.subheader(
            "My Progress"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Overall Mastery",
            f"{overview['overall_mastery']}%",
        )

        c2.metric(
            "Learning Support Indicator",
            (
                f"{overview['indicator']['level']} "
                f"({overview['indicator']['score']}/100)"
            ),
        )

        c3.metric(
            "Priority Topic",
            (
                overview[
                    "indicator"
                ][
                    "weakest_topic"
                ]
                or "N/A"
            ),
        )

        st.caption(
            "Prototype indicator only. "
            "This is not a validated predictor of "
            "academic failure or certification outcomes."
        )

        st.subheader(
            "Topic Mastery"
        )

        for (
            topic_name,
            topic_data,
        ) in overview["topics"].items():
            left, right = st.columns(
                [5, 1]
            )

            left.write(
                f"**{topic_name}**"
            )

            right.write(
                f"**{topic_data['mastery']}%**"
            )

            st.progress(
                max(
                    0.0,
                    min(
                        1.0,
                        topic_data[
                            "mastery"
                        ]
                        / 100,
                    ),
                )
            )

        if rec:
            st.subheader(
                "Recommended Next Step"
            )

            st.info(
                f"Focus on **{rec['topic']}**. "
                f"Current mastery: **{rec['mastery']}%**.\n\n"
                f"Review: **{rec['review']}**  \n"
                f"Practice: **{rec['practice']}**  \n"
                f"Target: **{rec['target']}%**"
            )

    # -----------------------------------------------------
    # PRACTICE QUIZ
    # -----------------------------------------------------

    elif page == "Practice Quiz":
        st.subheader(
            "Practice Quiz"
        )

        st.caption(
            "Prototype assessment recorder"
        )

        quiz_topic = st.selectbox(
            "Topic",
            [
                "Sketching",
                "Features",
                "Part Modeling",
                "Assemblies",
                "Drawings",
            ],
        )

        score = st.slider(
            "Quiz score",
            0,
            100,
            70,
        )

        attempts = st.number_input(
            "Attempts",
            1,
            10,
            1,
        )

        if st.button(
            "Record Quiz Result",
            type="primary",
        ):
            save_assessment(
                student_id,
                quiz_topic,
                "quiz",
                score,
                attempts,
            )

            st.success(
                "Quiz result recorded."
            )

    # -----------------------------------------------------
    # CERTIFICATION PROGRESS
    # -----------------------------------------------------

    else:
        overview = student_overview(
            student_id
        )

        readiness = (
            overview[
                "overall_mastery"
            ]
        )

        st.subheader(
            "CSWA Certification Progress"
        )

        c1, c2 = st.columns(
            [1, 2]
        )

        c1.metric(
            "Prototype Readiness",
            f"{readiness}%",
        )

        c2.progress(
            max(
                0.0,
                min(
                    1.0,
                    readiness / 100,
                ),
            )
        )

        st.caption(
            "Prototype only - this is not a validated "
            "predictor of CSWA exam performance."
        )


# =========================================================
# INSTRUCTOR PORTAL
# =========================================================

else:
    hero(
        "CSWAI",
        (
            "Instructor analytics and engineering "
            "knowledge management"
        ),
        "Instructor Portal",
    )

    (
        overview_tab,
        kb_tab,
        questions_tab,
        data_tab,
    ) = st.tabs(
        [
            "Class Overview",
            "Knowledge Base",
            "AI Questions",
            "Data",
        ]
    )

    # -----------------------------------------------------
    # CLASS OVERVIEW
    # -----------------------------------------------------

    with overview_tab:
        data = class_overview()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Students",
            len(
                data["students"]
            ),
        )

        c2.metric(
            "High Support Need",
            data["counts"]["High"],
        )

        c3.metric(
            "Medium Support Need",
            data["counts"]["Medium"],
        )

        c4.metric(
            "Weakest Class Topic",
            (
                data[
                    "weakest_class_topic"
                ]
                or "N/A"
            ),
        )

        if data[
            "topic_mastery"
        ]:
            st.subheader(
                "Class Topic Mastery"
            )

            chart_df = (
                pd.DataFrame(
                    {
                        "Topic": list(
                            data[
                                "topic_mastery"
                            ].keys()
                        ),
                        "Mastery": list(
                            data[
                                "topic_mastery"
                            ].values()
                        ),
                    }
                )
                .set_index(
                    "Topic"
                )
            )

            st.bar_chart(
                chart_df
            )

        # =============================================
        # NEW STUDENT LEARNING PRIORITY TABLE
        # =============================================

        st.subheader(
            "Student Learning Priorities"
        )

        st.caption(
            "Shows each student's current weakest topic, "
            "learning behavior, support level, and suggested "
            "next instructional action."
        )

        priority_df = pd.DataFrame(
            data["students"]
        )

        if not priority_df.empty:
            display_df = priority_df[
                [
                    "student_id",
                    "name",
                    "overall_mastery",
                    "needs_work_on",
                    "topic_mastery",
                    "attempts",
                    "ai_questions",
                    "support_level",
                    "support_score",
                    "recommended_action",
                ]
            ].copy()

            display_df.columns = [
                "Student ID",
                "Student",
                "Overall Mastery",
                "Needs Work On",
                "Topic Mastery",
                "Avg Attempts",
                "AI Questions",
                "Support Level",
                "Support Score",
                "Recommended Action",
            ]

            display_df = (
                display_df.sort_values(
                    [
                        "Support Score",
                        "Topic Mastery",
                    ],
                    ascending=[
                        False,
                        True,
                    ],
                )
            )

            st.dataframe(
                display_df,
                column_config={
                    "Overall Mastery":
                        st.column_config.ProgressColumn(
                            "Overall Mastery",
                            min_value=0,
                            max_value=100,
                            format="%.1f%%",
                        ),

                    "Topic Mastery":
                        st.column_config.ProgressColumn(
                            "Topic Mastery",
                            min_value=0,
                            max_value=100,
                            format="%.1f%%",
                        ),

                    "Avg Attempts":
                        st.column_config.NumberColumn(
                            "Avg Attempts",
                            format="%.1f",
                        ),

                    "AI Questions":
                        st.column_config.NumberColumn(
                            "AI Questions",
                            format="%d",
                        ),

                    "Support Score":
                        st.column_config.ProgressColumn(
                            "Support Score",
                            min_value=0,
                            max_value=100,
                            format="%d",
                        ),

                    "Recommended Action":
                        st.column_config.TextColumn(
                            "Recommended Action",
                            width="large",
                        ),
                },
                use_container_width=True,
                hide_index=True,
            )

        # =============================================
        # INDIVIDUAL STUDENT DETAIL
        # =============================================

        st.subheader(
            "Student Detail"
        )

        selected_student = (
            st.selectbox(
                "Select student",
                [
                    row[
                        "student_id"
                    ]
                    for row in data[
                        "students"
                    ]
                ],
                key="instructor_student_detail",
            )
        )

        selected_row = next(
            (
                row
                for row in data[
                    "students"
                ]
                if row[
                    "student_id"
                ]
                == selected_student
            ),
            None,
        )

        if selected_row:
            d1, d2, d3, d4 = (
                st.columns(4)
            )

            d1.metric(
                "Overall Mastery",
                f"{selected_row['overall_mastery']}%",
            )

            d2.metric(
                "Needs Work On",
                (
                    selected_row[
                        "needs_work_on"
                    ]
                    or "N/A"
                ),
            )

            d3.metric(
                "Topic Mastery",
                f"{selected_row['topic_mastery']}%",
            )

            d4.metric(
                "Support Level",
                selected_row[
                    "support_level"
                ],
            )

            st.markdown(
                "#### Why CSWAI identified this priority"
            )

            reasons = (
                selected_row[
                    "reasons"
                ].split(
                    " | "
                )
            )

            for reason in reasons:
                st.write(
                    f"- {reason}"
                )

            st.markdown(
                "#### Recommended Instructor Action"
            )

            st.info(
                selected_row[
                    "recommended_action"
                ]
            )

    # -----------------------------------------------------
    # KNOWLEDGE BASE
    # -----------------------------------------------------

    with kb_tab:
        kb = (
            knowledge_base_stats()
        )

        c1, c2, c3 = (
            st.columns(3)
        )

        c1.metric(
            "Documents",
            kb["documents"],
        )

        c2.metric(
            "Indexed Pages",
            kb["pages"],
        )

        c3.metric(
            "Knowledge Chunks",
            kb["chunks"],
        )

        st.subheader(
            "Knowledge Coverage"
        )

        left, right = (
            st.columns(2)
        )

        with left:
            st.markdown(
                "#### Topics"
            )

            if kb["topics"]:
                topic_df = (
                    pd.DataFrame(
                        [
                            {
                                "Topic": key,
                                "Chunks": value,
                            }
                            for key, value
                            in kb[
                                "topics"
                            ].items()
                        ]
                    )
                    .sort_values(
                        "Chunks",
                        ascending=False,
                    )
                )

                st.dataframe(
                    topic_df,
                    hide_index=True,
                    use_container_width=True,
                )

        with right:
            st.markdown(
                "#### EKE Knowledge Layers"
            )

            if kb[
                "eke_layers"
            ]:
                layer_df = (
                    pd.DataFrame(
                        [
                            {
                                "Layer": key,
                                "Chunks": value,
                            }
                            for key, value
                            in kb[
                                "eke_layers"
                            ].items()
                        ]
                    )
                    .sort_values(
                        "Chunks",
                        ascending=False,
                    )
                )

                st.dataframe(
                    layer_df,
                    hide_index=True,
                    use_container_width=True,
                )

        st.divider()

        st.markdown(
            "#### Knowledge Base Administration"
        )

        st.caption(
            "Rebuild only after adding or changing "
            "approved course documents."
        )

        if st.button(
            "Build / Rebuild Knowledge Index",
            type="primary",
        ):
            try:
                with st.spinner(
                    "Building the CSWAI knowledge index..."
                ):
                    count = build_index(
                        batch_size=8
                    )

                st.success(
                    "Knowledge base rebuilt successfully: "
                    f"{count} chunks."
                )

                st.rerun()

            except Exception as exc:
                st.error(
                    str(exc)
                )

    # -----------------------------------------------------
    # AI QUESTIONS
    # -----------------------------------------------------

    with questions_tab:
        st.subheader(
            "Recent Student Questions"
        )

        conn = get_connection()

        qdf = pd.read_sql_query(
            """
            SELECT
                student_id,
                topic,
                question,
                timestamp
            FROM interactions
            ORDER BY id DESC
            LIMIT 50
            """,
            conn,
        )

        conn.close()

        st.dataframe(
            qdf,
            use_container_width=True,
            hide_index=True,
        )

        if not qdf.empty:
            st.subheader(
                "Question Topics"
            )

            question_counts = (
                qdf[
                    "topic"
                ]
                .value_counts()
                .rename_axis(
                    "Topic"
                )
                .to_frame(
                    "Questions"
                )
            )

            st.bar_chart(
                question_counts
            )

    # -----------------------------------------------------
    # DATABASE VIEW
    # -----------------------------------------------------

    with data_tab:
        st.subheader(
            "Database Status"
        )

        conn = get_connection()

        student_count = (
            conn.execute(
                "SELECT COUNT(*) FROM students"
            ).fetchone()[0]
        )

        assessment_count = (
            conn.execute(
                "SELECT COUNT(*) FROM assessments"
            ).fetchone()[0]
        )

        interaction_count = (
            conn.execute(
                "SELECT COUNT(*) FROM interactions"
            ).fetchone()[0]
        )

        d1, d2, d3, d4 = (
            st.columns(4)
        )

        d1.metric(
            "Students",
            student_count,
        )

        d2.metric(
            "Assessments",
            assessment_count,
        )

        d3.metric(
            "AI Interactions",
            interaction_count,
        )

        d4.metric(
            "Data Type",
            "Synthetic Demo",
        )

        st.warning(
            "The current assessment records are synthetic "
            "prototype data. They are not real student outcomes."
        )

        students_df = (
            pd.read_sql_query(
                """
                SELECT *
                FROM students
                ORDER BY student_id
                """,
                conn,
            )
        )

        assessments_df = (
            pd.read_sql_query(
                """
                SELECT *
                FROM assessments
                ORDER BY id DESC
                """,
                conn,
            )
        )

        interactions_df = (
            pd.read_sql_query(
                """
                SELECT
                    id,
                    student_id,
                    topic,
                    question,
                    timestamp
                FROM interactions
                ORDER BY id DESC
                """,
                conn,
            )
        )

        conn.close()

        raw_students, raw_assessments, raw_interactions = (
            st.tabs(
                [
                    "Students",
                    "Assessments",
                    "Interactions",
                ]
            )
        )

        with raw_students:
            st.dataframe(
                students_df,
                use_container_width=True,
                hide_index=True,
            )

        with raw_assessments:
            st.dataframe(
                assessments_df,
                use_container_width=True,
                hide_index=True,
            )

        with raw_interactions:
            st.dataframe(
                interactions_df,
                use_container_width=True,
                hide_index=True,
            )
