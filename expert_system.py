"""
Expert System Module: Rule-Based Academic Recommendation Engine
Provides deterministic, pedagogical, and actionable guidance for students
based on academic history, study habits, lifestyle patterns, and model predictions.
"""

def generate_recommendations(student_data: dict, predicted_g3: float = None, is_at_risk: bool = None) -> dict:
    """
    Evaluates student attributes and ML predictions using an expert rule-base.
    
    Parameters:
        student_data (dict): Dictionary of student features (studytime, failures, absences, etc.)
        predicted_g3 (float, optional): Model predicted final grade (0-20)
        is_at_risk (bool, optional): Whether classified as 'At Risk' (G3 < 10)
        
    Returns:
        dict: Structured recommendation dossier including urgency, alerts, action items, and schedule.
    """
    # Extract key features with sensible fallbacks
    studytime = int(student_data.get('studytime', 2))
    failures = int(student_data.get('failures', 0))
    absences = int(student_data.get('absences', 0))
    g1 = float(student_data.get('G1', 10))
    g2 = float(student_data.get('G2', 10))
    dalc = int(student_data.get('Dalc', 1))
    walc = int(student_data.get('Walc', 1))
    goout = int(student_data.get('goout', 3))
    schoolsup = str(student_data.get('schoolsup', 'no')).lower()
    famsup = str(student_data.get('famsup', 'no')).lower()
    paid = str(student_data.get('paid', 'no')).lower()
    internet = str(student_data.get('internet', 'yes')).lower()
    higher = str(student_data.get('higher', 'yes')).lower()

    if predicted_g3 is None:
        predicted_g3 = (g1 + g2) / 2.0
        
    if is_at_risk is None:
        is_at_risk = predicted_g3 < 10.0

    # Containers for recommendations
    academic_actions = []
    attendance_actions = []
    lifestyle_actions = []
    support_actions = []
    strengths = []
    warnings = []

    # ------------------ RULE SET 1: Academic Trajectory & Risk ------------------
    if is_at_risk or predicted_g3 < 10.0:
        urgency = "HIGH RISK - IMMEDIATE INTERVENTION REQUIRED"
        urgency_level = "danger"
        warnings.append(f"Predicted Final Grade ({predicted_g3:.1f}/20) is below the minimum passing threshold (10.0).")
        academic_actions.append({
            "title": "Mandatory Remedial Classes",
            "priority": "Critical",
            "detail": "Enroll immediately in remedial mathematics tutorial sessions 3 days/week focusing on foundational algebra and calculus."
        })
        academic_actions.append({
            "title": "Diagnostic Test Review",
            "priority": "High",
            "detail": f"Review G1 ({g1:.0f}) and G2 ({g2:.0f}) examination mistakes with the course instructor within the next 7 days."
        })
    elif predicted_g3 < 14.0:
        urgency = "MODERATE NEED - PERFORMANCE ENHANCEMENT"
        urgency_level = "warning"
        strengths.append(f"Predicted Final Grade ({predicted_g3:.1f}/20) is currently in the passing bracket.")
        academic_actions.append({
            "title": "Targeted Problem-Solving Drills",
            "priority": "Medium",
            "detail": "Solve at least 5 previous year semester exam papers under timed conditions to raise score into distinction range (14+)."
        })
    else:
        urgency = "EXCELLENCE - ADVANCED ENRICHMENT"
        urgency_level = "success"
        strengths.append(f"Strong Academic Trajectory! Predicted Final Grade is {predicted_g3:.1f}/20.")
        academic_actions.append({
            "title": "Honors & Advanced Problem Solving",
            "priority": "Low",
            "detail": "Engage in competitive mathematics contests, peer-mentoring, and advanced STEM project electives."
        })

    # Grade Trend Analysis
    if g2 < g1:
        warnings.append(f"Performance declined from Term 1 (G1 = {g1:.0f}) to Term 2 (G2 = {g2:.0f}).")
        academic_actions.append({
            "title": "Coursework Recovery Strategy",
            "priority": "High",
            "detail": "Identify topics where understanding dropped between Term 1 and Term 2; schedule targeted faculty office hours."
        })
    elif g2 > g1:
        strengths.append(f"Positive momentum detected: Grade improved from Term 1 (G1 = {g1:.0f}) to Term 2 (G2 = {g2:.0f}).")

    # ------------------ RULE SET 2: Study Time Discipline ------------------
    # 1: <2 hrs, 2: 2-5 hrs, 3: 5-10 hrs, 4: >10 hrs
    if studytime == 1:
        warnings.append("Insufficient study time (< 2 hours/week). High correlation with academic failure.")
        academic_actions.append({
            "title": "Increase Weekly Study Hours",
            "priority": "Critical",
            "detail": "Increase study commitment from <2 hrs to at least 6-8 hours weekly. Implement 45-minute daily focus blocks using Pomodoro technique."
        })
    elif studytime == 2:
        academic_actions.append({
            "title": "Optimize Study Routines",
            "priority": "Medium",
            "detail": "Increase weekly revision time to 5-10 hours, emphasizing active recall and practice problems over passive note reading."
        })
    else:
        strengths.append(f"Dedicated study discipline: Invests {'>10 hours' if studytime == 4 else '5-10 hours'} weekly.")

    # ------------------ RULE SET 3: Failures History ------------------
    if failures > 0:
        warnings.append(f"Student has a history of {failures} previous course failure(s).")
        academic_actions.append({
            "title": "Core Concept Remediation",
            "priority": "High",
            "detail": "Conduct an individual learning audit with a mentor to clear historical concept gaps from prerequisite subjects."
        })

    # ------------------ RULE SET 4: Attendance & Absenteeism ------------------
    if absences >= 15:
        warnings.append(f"Severe absenteeism alert: {absences} missed school days.")
        attendance_actions.append({
            "title": "Urgent Attendance Contract",
            "priority": "Critical",
            "detail": "Commit to 95%+ attendance for all remaining lectures. Missed classes directly lower final grade probability."
        })
        attendance_actions.append({
            "title": "Lecture Notes Audit",
            "priority": "High",
            "detail": "Borrow and transcribe missed lecture notes from high-performing peers; submit missed assignments promptly."
        })
    elif absences >= 8:
        warnings.append(f"Elevated absenteeism ({absences} days missed).")
        attendance_actions.append({
            "title": "Attendance Monitoring",
            "priority": "Medium",
            "detail": "Set a goal of zero unexcused absences before the final examinations."
        })
    else:
        strengths.append(f"Consistent class attendance record ({absences} absences).")

    # ------------------ RULE SET 5: Lifestyle, Social Life & Health ------------------
    if walc >= 4 or dalc >= 3:
        warnings.append("High alcohol consumption reported (Dalc/Walc), significantly impacting cognitive retention and sleep.")
        lifestyle_actions.append({
            "title": "Lifestyle & Wellness Counseling",
            "priority": "High",
            "detail": "Consult campus health wellness counselors for guidance on moderation and stress coping mechanisms."
        })

    if goout >= 4 and studytime <= 2:
        warnings.append("Imbalance between socializing (goout >= 4) and study commitment.")
        lifestyle_actions.append({
            "title": "Social & Study Schedule Rebalance",
            "priority": "Medium",
            "detail": "Limit evening outings to Saturday nights only until term examinations conclude."
        })

    # ------------------ RULE SET 6: Institutional & Family Resources ------------------
    if schoolsup == 'no' and is_at_risk:
        support_actions.append({
            "title": "Apply for Institutional Tutoring",
            "priority": "High",
            "detail": "Submit an application to the department academic coordinator for free peer mentoring and tutoring assistance."
        })

    if paid == 'no' and is_at_risk:
        support_actions.append({
            "title": "Access Free Digital Tutoring Resources",
            "priority": "Medium",
            "detail": "Leverage open educational resources (Khan Academy, MIT OpenCourseWare, YouTube STEM lectures)."
        })

    if internet == 'no':
        support_actions.append({
            "title": "Utilize Campus Digital Infrastructure",
            "priority": "Medium",
            "detail": "Schedule study sessions in the campus computer lab / university library to access online learning repositories."
        })

    if famsup == 'no':
        support_actions.append({
            "title": "Form a Peer Study Cohort",
            "priority": "Medium",
            "detail": "Join or create a study circle with 2-3 classmates for mutual accountability and collaborative exam review."
        })

    if higher == 'yes':
        strengths.append("High intrinsic motivation: Aspirations to pursue higher tertiary education.")

    # Recommended weekly timetable allocation
    recommended_weekly_hours = max(8, 12 if is_at_risk else (6 if predicted_g3 >= 14 else 9))
    weekly_schedule = {
        "Monday": "1.5 hrs: Core mathematics lecture review & notes synthesis",
        "Tuesday": "1.5 hrs: Practice problem sets & homework assignments",
        "Wednesday": "1.5 hrs: Difficult topic deep-dive & teacher office hours consultation",
        "Thursday": "1.5 hrs: Peer study group discussion & concept cross-testing",
        "Friday": "1.0 hr: Weekly summary flashcards & self-assessment quiz",
        "Saturday": f"{2.5 if is_at_risk else 1.5} hrs: Timed mock test paper & error analysis",
        "Sunday": "1.0 hr: Review weekly progress, rest & plan next week's schedule",
        "Total Target Weekly Hours": f"{recommended_weekly_hours} Hours/Week"
    }

    return {
        "urgency": urgency,
        "urgency_level": urgency_level,
        "predicted_g3": round(predicted_g3, 2),
        "is_at_risk": is_at_risk,
        "strengths": strengths if strengths else ["Basic class enrollment maintained."],
        "warnings": warnings if warnings else ["No acute risk factors identified."],
        "academic_actions": academic_actions,
        "attendance_actions": attendance_actions,
        "lifestyle_actions": lifestyle_actions,
        "support_actions": support_actions,
        "weekly_schedule": weekly_schedule
    }
