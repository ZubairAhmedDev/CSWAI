from .db import get_connection


# =========================================================
# LEARNING RESOURCES
# =========================================================

RESOURCES = {
    "Sketching": (
        "Sketching Fundamentals",
        "Sketch Practice Exercise 2",
    ),
    "Features": (
        "Feature Modeling Review",
        "Feature Practice Exercise 3",
    ),
    "Part Modeling": (
        "Part Modeling Workflow Review",
        "Part Modeling Practice Exercise 3",
    ),
    "Assemblies": (
        "Assembly Mates Review",
        "Assembly Practice Exercise 4",
    ),
    "Drawings": (
        "Engineering Drawing Review",
        "Drawing Practice Exercise 2",
    ),
}


# =========================================================
# STUDENT TOPIC STATISTICS
# =========================================================

def topic_stats(student_id):
    conn = get_connection()

    assessment_rows = conn.execute(
        """
        SELECT
            topic,
            AVG(score) AS mastery,
            AVG(attempts) AS attempts
        FROM assessments
        WHERE student_id = ?
        GROUP BY topic
        """,
        (student_id,),
    ).fetchall()

    question_rows = conn.execute(
        """
        SELECT
            topic,
            COUNT(*) AS qcount
        FROM interactions
        WHERE student_id = ?
        GROUP BY topic
        """,
        (student_id,),
    ).fetchall()

    conn.close()

    question_map = {
        row["topic"]: row["qcount"]
        for row in question_rows
    }

    stats = {}

    for row in assessment_rows:
        stats[row["topic"]] = {
            "mastery": round(row["mastery"], 1),
            "attempts": round(row["attempts"], 1),
            "question_count": question_map.get(row["topic"], 0),
        }

    return stats


# =========================================================
# LEARNING SUPPORT INDICATOR
# =========================================================

def support_indicator(student_id):
    stats = topic_stats(student_id)

    if not stats:
        return {
            "score": 0,
            "level": "Low",
            "weakest_topic": None,
            "reasons": ["Not enough learning data yet"],
        }

    weakest_topic = min(
        stats,
        key=lambda topic: stats[topic]["mastery"],
    )

    weakest_data = stats[weakest_topic]

    score = 0
    reasons = []

    # Low mastery
    if weakest_data["mastery"] < 50:
        score += 45
        reasons.append(
            f"{weakest_topic} mastery is below 50% "
            f"({weakest_data['mastery']}%)"
        )

    elif weakest_data["mastery"] < 70:
        score += 25
        reasons.append(
            f"{weakest_topic} mastery is below 70% "
            f"({weakest_data['mastery']}%)"
        )

    elif weakest_data["mastery"] < 80:
        score += 10
        reasons.append(
            f"{weakest_topic} could benefit from additional review "
            f"({weakest_data['mastery']}%)"
        )

    # Repeated attempts
    if weakest_data["attempts"] >= 3:
        score += 25
        reasons.append(
            f"Repeated attempts in {weakest_topic} "
            f"({weakest_data['attempts']:.1f} average attempts)"
        )

    elif weakest_data["attempts"] >= 2:
        score += 12
        reasons.append(
            f"Multiple attempts in {weakest_topic}"
        )

    # Repeated AI help-seeking
    if weakest_data["question_count"] >= 6:
        score += 20
        reasons.append(
            f"Frequent CSWAI questions about {weakest_topic} "
            f"({weakest_data['question_count']} questions)"
        )

    elif weakest_data["question_count"] >= 3:
        score += 10
        reasons.append(
            f"Repeated CSWAI questions about {weakest_topic} "
            f"({weakest_data['question_count']} questions)"
        )

    # Overall mastery
    overall_mastery = (
        sum(
            topic_data["mastery"]
            for topic_data in stats.values()
        )
        / len(stats)
    )

    if overall_mastery < 60:
        score += 10
        reasons.append(
            f"Overall mastery is low ({overall_mastery:.1f}%)"
        )

    elif overall_mastery < 70:
        score += 5

    score = min(100, round(score))

    if score >= 60:
        level = "High"
    elif score >= 30:
        level = "Medium"
    else:
        level = "Low"

    return {
        "score": score,
        "level": level,
        "weakest_topic": weakest_topic,
        "reasons": (
            reasons
            or ["Current indicators are within expected range"]
        ),
    }


# =========================================================
# STUDENT OVERVIEW
# =========================================================

def student_overview(student_id):
    stats = topic_stats(student_id)

    if stats:
        overall_mastery = round(
            sum(
                topic_data["mastery"]
                for topic_data in stats.values()
            )
            / len(stats),
            1,
        )
    else:
        overall_mastery = 0

    return {
        "overall_mastery": overall_mastery,
        "topics": stats,
        "indicator": support_indicator(student_id),
    }


# =========================================================
# PERSONALIZED RECOMMENDATION
# =========================================================

def recommendation(student_id):
    overview = student_overview(student_id)

    weakest_topic = overview["indicator"]["weakest_topic"]

    if not weakest_topic:
        return None

    review, practice = RESOURCES.get(
        weakest_topic,
        (
            "Instructor-approved review material",
            "Instructor-approved practice activity",
        ),
    )

    topic_data = overview["topics"][weakest_topic]

    return {
        "topic": weakest_topic,
        "mastery": topic_data["mastery"],
        "attempts": topic_data["attempts"],
        "question_count": topic_data["question_count"],
        "review": review,
        "practice": practice,
        "target": 70,
    }


# =========================================================
# CLASS / INSTRUCTOR OVERVIEW
# =========================================================

def class_overview():
    conn = get_connection()

    students = conn.execute(
        """
        SELECT
            student_id,
            name
        FROM students
        ORDER BY student_id
        """
    ).fetchall()

    conn.close()

    student_rows = []
    topic_accumulator = {}

    counts = {
        "Low": 0,
        "Medium": 0,
        "High": 0,
    }

    for student in students:
        student_id = student["student_id"]

        overview = student_overview(student_id)
        indicator = overview["indicator"]

        weakest_topic = indicator["weakest_topic"]

        weakest_mastery = 0
        weakest_attempts = 0
        weakest_questions = 0

        if (
            weakest_topic
            and weakest_topic in overview["topics"]
        ):
            weak_data = overview["topics"][weakest_topic]

            weakest_mastery = weak_data["mastery"]
            weakest_attempts = weak_data["attempts"]
            weakest_questions = weak_data["question_count"]

        rec = recommendation(student_id)

        if rec:
            recommended_action = (
                f"Review {rec['review']} then complete "
                f"{rec['practice']}"
            )
        else:
            recommended_action = (
                "Continue current learning plan"
            )

        counts[indicator["level"]] += 1

        student_rows.append(
            {
                "student_id": student_id,
                "name": student["name"],
                "overall_mastery": overview["overall_mastery"],
                "needs_work_on": weakest_topic,
                "topic_mastery": weakest_mastery,
                "attempts": weakest_attempts,
                "ai_questions": weakest_questions,
                "support_level": indicator["level"],
                "support_score": indicator["score"],
                "reasons": " | ".join(indicator["reasons"]),
                "recommended_action": recommended_action,
            }
        )

        for topic, topic_data in overview["topics"].items():
            topic_accumulator.setdefault(topic, []).append(
                topic_data["mastery"]
            )

    topic_mastery = {
        topic: round(
            sum(values) / len(values),
            1,
        )
        for topic, values in topic_accumulator.items()
    }

    weakest_class_topic = (
        min(
            topic_mastery,
            key=topic_mastery.get,
        )
        if topic_mastery
        else None
    )

    return {
        "counts": counts,
        "students": student_rows,
        "topic_mastery": topic_mastery,
        "weakest_class_topic": weakest_class_topic,
    }
