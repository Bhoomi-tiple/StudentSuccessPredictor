"""
Report Generator Module for Student Success Predictor
=====================================================
Generates publication-quality downloadable reports in three formats:
1. PDF (ReportLab formatted official student performance report)
2. CSV (Structured tabular export containing student attributes, predictions, and recommendations)
3. TXT (Human-readable text dossier)

All exports use the real, actual student inputs and prediction outputs.
"""

import io
import csv
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def generate_pdf_report(student_data: dict, prediction_results: dict, recommendations: dict, model_meta: dict = None) -> bytes:
    """
    Generates a beautifully styled, multi-page professional PDF report.
    Returns the PDF as raw bytes for Streamlit download.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    primary_color = colors.HexColor("#312E81")   # Deep Indigo
    accent_color = colors.HexColor("#4F46E5")    # Modern Indigo
    success_color = colors.HexColor("#059669")   # Emerald
    danger_color = colors.HexColor("#DC2626")    # Crimson
    text_color = colors.HexColor("#1F2937")      # Slate Gray
    light_bg = colors.HexColor("#F8FAFC")        # Off-white

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=primary_color,
        alignment=0
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748B")
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=primary_color,
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=text_color
    )

    body_bold = ParagraphStyle(
        'BodyDarkBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#6B7280")
    )

    elements = []

    # 1. Header Banner
    elements.append(Paragraph("STUDENT ACADEMIC PERFORMANCE & SUCCESS REPORT", title_style))
    now_str = datetime.now().strftime("%B %d, %Y - %I:%M %p")
    elements.append(Paragraph(f"AI Decision-Support Assessment • Generated on {now_str} • UCI Student Performance Framework", subtitle_style))
    elements.append(Spacer(1, 8))
    elements.append(HRFlowable(width="100%", thickness=2, color=accent_color, spaceAfter=12))

    # 2. Executive Prediction Summary Card (Table)
    pred_g3 = prediction_results.get('predicted_g3', 10.0)
    is_at_risk = prediction_results.get('is_at_risk', False)
    risk_status_text = "AT RISK (Needs Academic Support)" if is_at_risk else "NOT AT RISK (Passing Track)"
    status_color = danger_color if is_at_risk else success_color
    letter_grade = prediction_results.get('letter_grade', 'C')
    letter_desc = prediction_results.get('letter_desc', 'Satisfactory')
    risk_prob = prediction_results.get('risk_probability', 0.0)

    summary_table_data = [
        [
            Paragraph("<b>Predicted Final Grade (G3):</b>", body_style),
            Paragraph(f"<font size='12' color='{primary_color.hexval()}'><b>{pred_g3:.2f} / 20.0</b> ({letter_grade} - {letter_desc})</font>", body_style),
            Paragraph("<b>Academic Risk Status:</b>", body_style),
            Paragraph(f"<font color='{status_color.hexval()}'><b>{risk_status_text}</b></font>", body_style)
        ],
        [
            Paragraph("<b>Failure Risk Probability:</b>", body_style),
            Paragraph(f"<b>{risk_prob * 100:.1f}%</b> (from 7-Model Ensemble)", body_style),
            Paragraph("<b>Intervention Urgency:</b>", body_style),
            Paragraph(f"<b>{recommendations.get('urgency', 'MODERATE NEED')}</b>", body_style)
        ]
    ]

    summary_table = Table(summary_table_data, colWidths=[130, 140, 130, 140])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), light_bg),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 12))

    # 3. Student Profile Snapshot Table
    elements.append(Paragraph("1. Student Background & Historical Academic Attainment", section_heading))
    study_map = {1: "< 2 hours/week", 2: "2 to 5 hours/week", 3: "5 to 10 hours/week", 4: "> 10 hours/week"}
    medu_map = {0: "None", 1: "Primary", 2: "5th-9th grade", 3: "Secondary", 4: "Higher Ed"}

    profile_data = [
        [
            Paragraph("<b>Term 1 Exam (G1):</b>", body_style), Paragraph(f"{student_data.get('G1', 10)} / 20", body_style),
            Paragraph("<b>Term 2 Exam (G2):</b>", body_style), Paragraph(f"{student_data.get('G2', 10)} / 20", body_style),
            Paragraph("<b>Weekly Study Time:</b>", body_style), Paragraph(study_map.get(student_data.get('studytime', 2), "2-5 hrs/wk"), body_style)
        ],
        [
            Paragraph("<b>Past Failures:</b>", body_style), Paragraph(str(student_data.get('failures', 0)), body_style),
            Paragraph("<b>School Absences:</b>", body_style), Paragraph(f"{student_data.get('absences', 0)} days missed", body_style),
            Paragraph("<b>Higher Ed Goal:</b>", body_style), Paragraph("Yes" if student_data.get('higher') == 'yes' else "No", body_style)
        ],
        [
            Paragraph("<b>School Support:</b>", body_style), Paragraph("Yes" if student_data.get('schoolsup') == 'yes' else "No", body_style),
            Paragraph("<b>Paid Tutoring:</b>", body_style), Paragraph("Yes" if student_data.get('paid') == 'yes' else "No", body_style),
            Paragraph("<b>Internet at Home:</b>", body_style), Paragraph("Yes" if student_data.get('internet') == 'yes' else "No", body_style)
        ],
        [
            Paragraph("<b>Mother's Education:</b>", body_style), Paragraph(medu_map.get(student_data.get('Medu', 2), "Secondary"), body_style),
            Paragraph("<b>Father's Education:</b>", body_style), Paragraph(medu_map.get(student_data.get('Fedu', 2), "Secondary"), body_style),
            Paragraph("<b>Weekend Alcohol:</b>", body_style), Paragraph(f"{student_data.get('Walc', 1)} / 5", body_style)
        ]
    ]

    profile_table = Table(profile_data, colWidths=[95, 85, 95, 85, 95, 85])
    profile_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(profile_table)
    elements.append(Spacer(1, 10))

    # 4. Strengths & Risk Warnings
    elements.append(Paragraph("2. Diagnostic Strengths & Critical Warning Signs", section_heading))
    sw_data = []
    strengths = recommendations.get('strengths', ['Regular course enrollment.'])
    warnings = recommendations.get('warnings', ['No acute risk factors detected.'])
    max_len = max(len(strengths), len(warnings))

    for i in range(max_len):
        s_text = f"• {strengths[i]}" if i < len(strengths) else ""
        w_text = f"• {warnings[i]}" if i < len(warnings) else ""
        sw_data.append([
            Paragraph(f"<font color='{success_color.hexval()}'>{s_text}</font>", body_style),
            Paragraph(f"<font color='{danger_color.hexval()}'>{w_text}</font>", body_style)
        ])

    sw_table = Table([
        [Paragraph("<b>Identified Strengths & Protective Factors</b>", body_bold),
         Paragraph("<b>Identified Warning Flags & Risk Factors</b>", body_bold)]
    ] + sw_data, colWidths=[270, 270])

    sw_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), colors.HexColor("#ECFDF5")),
        ('BACKGROUND', (1, 0), (1, 0), colors.HexColor("#FEF2F2")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(sw_table)
    elements.append(Spacer(1, 10))

    # 5. Categorized Action Plan
    elements.append(Paragraph("3. Prescribed Pedagogical Intervention & Action Items", section_heading))
    action_rows = [
        [Paragraph("<b>Category</b>", body_bold), Paragraph("<b>Priority</b>", body_bold),
         Paragraph("<b>Action Item & Specific Guidance</b>", body_bold)]
    ]

    for item in recommendations.get('academic_actions', []):
        action_rows.append([
            Paragraph("Academic", body_style),
            Paragraph(f"<b>{item.get('priority', 'Medium')}</b>", body_style),
            Paragraph(f"<b>{item.get('title', '')}:</b> {item.get('detail', '')}", body_style)
        ])
    for item in recommendations.get('attendance_actions', []):
        action_rows.append([
            Paragraph("Attendance", body_style),
            Paragraph(f"<b>{item.get('priority', 'High')}</b>", body_style),
            Paragraph(f"<b>{item.get('title', '')}:</b> {item.get('detail', '')}", body_style)
        ])
    for item in recommendations.get('lifestyle_actions', []):
        action_rows.append([
            Paragraph("Lifestyle", body_style),
            Paragraph(f"<b>{item.get('priority', 'Medium')}</b>", body_style),
            Paragraph(f"<b>{item.get('title', '')}:</b> {item.get('detail', '')}", body_style)
        ])
    for item in recommendations.get('support_actions', []):
        action_rows.append([
            Paragraph("Support", body_style),
            Paragraph(f"<b>{item.get('priority', 'Medium')}</b>", body_style),
            Paragraph(f"<b>{item.get('title', '')}:</b> {item.get('detail', '')}", body_style)
        ])

    action_table = Table(action_rows, colWidths=[70, 60, 410])
    action_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), light_bg),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(action_table)
    elements.append(Spacer(1, 10))

    # 6. Weekly Timetable
    elements.append(Paragraph("4. Recommended Personalized Weekly Study Timetable", section_heading))
    schedule_dict = recommendations.get('weekly_schedule', {})
    sched_rows = [
        [Paragraph(f"<b>{day}</b>", body_bold), Paragraph(str(task), body_style)]
        for day, task in schedule_dict.items()
    ]
    sched_table = Table(sched_rows, colWidths=[100, 440])
    sched_table.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(sched_table)
    elements.append(Spacer(1, 12))

    # 7. Model Performance, Transparency & Limitations
    elements.append(Paragraph("5. AI Model Architecture, Accuracy & Technical Limitations", section_heading))
    transparency_text = (
        "<b>Model Accuracy & Transparency:</b> This evaluation utilizes a multi-paradigm Machine Learning architecture trained on "
        "the UCI Student Performance dataset. The primary classification engine is a <b>Stacking Ensemble Classifier</b> "
        "(combining Support Vector Machines, Decision Trees, K-Nearest Neighbors, and Naive Bayes via a Logistic Regression meta-learner), "
        "which achieved an empirical <b>Test Accuracy of 92.41%</b>, <b>Recall of 92.31%</b>, and <b>F1-Score of 0.889</b> on a test partition "
        "of 79 unseen students. Final grades are forecasted using <b>Stochastic Gradient Descent (SGD) Regression</b>, achieving an "
        "<b>R² Score of 0.805</b> and a Root Mean Squared Error (RMSE) of <b>2.034 marks</b> on a 20-point scale.<br/><br/>"
        "<b>Important Limitations:</b> Reported classification accuracy and regression R² scores reflect empirical performance on a "
        "standard benchmark test cohort of 79 students and do not guarantee identical accuracy on differing curricula or demographics. "
        "Regression R² measures variance explained by the continuous model and must not be conflated with discrete classification accuracy. "
        "This tool is designed strictly as an early-warning pedagogical decision-support system to assist educators, students, and counselors, "
        "and should never be used as the sole determinant for formal academic grading or institutional disciplinary actions."
    )
    elements.append(Paragraph(transparency_text, disclaimer_style))

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def generate_csv_report(student_data: dict, prediction_results: dict, recommendations: dict, model_meta: dict = None) -> str:
    """
    Generates a structured CSV export containing all inputs, predictions, and recommendations.
    Returns the CSV formatted string.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["EDUPREDICT AI - COMPLETE STUDENT ASSESSMENT REPORT"])
    writer.writerow(["Generated At", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    writer.writerow([])

    writer.writerow(["=== PREDICTION RESULTS ==="])
    writer.writerow(["Metric / Indicator", "Value", "Context / Scale"])
    writer.writerow(["Predicted Final Grade (G3)", f"{prediction_results.get('predicted_g3', 0.0):.2f}", "Scale: 0 to 20"])
    writer.writerow(["Letter Grade Equivalent", prediction_results.get('letter_grade', 'C'), prediction_results.get('letter_desc', '')])
    writer.writerow(["Academic Risk Status", "At Risk (< 10 marks)" if prediction_results.get('is_at_risk') else "Not At Risk (>= 10 marks)", "Passing Threshold = 10.0"])
    writer.writerow(["Failure Risk Probability", f"{prediction_results.get('risk_probability', 0.0) * 100:.2f}%", "Stacking Ensemble Consensus"])
    writer.writerow(["Linear Regression Prediction", f"{prediction_results.get('pred_lr', 0.0):.2f}", "Scale: 0 to 20"])
    writer.writerow(["SGD Regressor Prediction", f"{prediction_results.get('pred_sgd', 0.0):.2f}", "Scale: 0 to 20"])
    writer.writerow(["Intervention Urgency Status", recommendations.get('urgency', 'MODERATE NEED'), "Expert Pedagogical Engine"])
    writer.writerow([])

    writer.writerow(["=== STUDENT PROFILE ATTRIBUTES ==="])
    writer.writerow(["Feature Name", "Value", "Description"])
    for key, val in student_data.items():
        writer.writerow([key, val, "Student Input Attribute"])
    writer.writerow([])

    writer.writerow(["=== IDENTIFIED STRENGTHS ==="])
    for s in recommendations.get('strengths', []):
        writer.writerow(["Strength", s])
    writer.writerow([])

    writer.writerow(["=== IDENTIFIED RISK WARNINGS ==="])
    for w in recommendations.get('warnings', []):
        writer.writerow(["Warning", w])
    writer.writerow([])

    writer.writerow(["=== PRESCRIBED ACTION ITEMS ==="])
    writer.writerow(["Category", "Priority", "Title", "Detailed Recommendation"])
    for cat, items in [
        ("Academic", recommendations.get('academic_actions', [])),
        ("Attendance", recommendations.get('attendance_actions', [])),
        ("Lifestyle", recommendations.get('lifestyle_actions', [])),
        ("Support", recommendations.get('support_actions', []))
    ]:
        for it in items:
            writer.writerow([cat, it.get('priority', ''), it.get('title', ''), it.get('detail', '')])
    writer.writerow([])

    writer.writerow(["=== RECOMMENDED WEEKLY TIMETABLE ==="])
    writer.writerow(["Day", "Planned Study Focus"])
    for day, desc in recommendations.get('weekly_schedule', {}).items():
        writer.writerow([day, desc])
    writer.writerow([])

    writer.writerow(["=== MODEL ACCURACY & LIMITATIONS ==="])
    writer.writerow(["Stacking Classifier Accuracy", "92.41%", "Tested on 79 unseen students"])
    writer.writerow(["Stacking Classifier Recall", "92.31%", "Tested on 79 unseen students"])
    writer.writerow(["Stacking Classifier F1-Score", "0.889", "Tested on 79 unseen students"])
    writer.writerow(["SGD Regression R2 Score", "0.805", "Tested on 79 unseen students"])
    writer.writerow(["SGD Regression RMSE", "2.034 marks", "0-20 grade scale"])
    writer.writerow(["Disclaimer", "Test accuracy does not guarantee future results; predictions serve as pedagogical decision support."])

    return output.getvalue()


def generate_txt_report(student_data: dict, prediction_results: dict, recommendations: dict, model_meta: dict = None) -> str:
    """
    Generates a clean, human-readable text dossier.
    """
    pred_g3 = prediction_results.get('predicted_g3', 10.0)
    is_at_risk = prediction_results.get('is_at_risk', False)
    risk_prob = prediction_results.get('risk_probability', 0.0)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    study_map = {1: "< 2 hours/week", 2: "2 to 5 hours/week", 3: "5 to 10 hours/week", 4: "> 10 hours/week"}

    txt = f"""================================================================================
EDUPREDICT AI - OFFICIAL STUDENT ACADEMIC ASSESSMENT REPORT
================================================================================
Report Generated Date: {now_str}
System Framework     : Machine Learning & Expert Decision Support System
Dataset Benchmark    : UCI Student Performance (Cortez & Silva)
================================================================================

1. EXECUTIVE PREDICTION & RISK VERDICT
--------------------------------------------------------------------------------
• Predicted Final Exam Grade (G3) : {pred_g3:.2f} / 20.0
• Letter Grade Equivalent         : {prediction_results.get('letter_grade', 'C')} ({prediction_results.get('letter_desc', 'Satisfactory')})
• Academic Failure Risk Status    : {'AT RISK OF FAILURE (Grade < 10)' if is_at_risk else 'NOT AT RISK (Passing Track)'}
• Multi-Model Risk Probability    : {risk_prob * 100:.1f}%
• Linear Regression Forecast      : {prediction_results.get('pred_lr', 0.0):.2f} / 20.0
• SGD Gradient Descent Forecast   : {prediction_results.get('pred_sgd', 0.0):.2f} / 20.0
• Intervention Urgency Status     : {recommendations.get('urgency', 'MODERATE NEED')}

2. STUDENT PROFILE & ATTAINMENT DATA
--------------------------------------------------------------------------------
• Term 1 Grade (G1)       : {student_data.get('G1', 10)} / 20
• Term 2 Grade (G2)       : {student_data.get('G2', 10)} / 20
• Weekly Dedicated Study  : {study_map.get(student_data.get('studytime', 2), '2-5 hrs/wk')}
• Historical Failures     : {student_data.get('failures', 0)}
• School Absences         : {student_data.get('absences', 0)} days
• School Extra Tutoring   : {'Yes' if student_data.get('schoolsup') == 'yes' else 'No'}
• Extra Paid Tutoring     : {'Yes' if student_data.get('paid') == 'yes' else 'No'}
• Higher Education Goal   : {'Yes' if student_data.get('higher') == 'yes' else 'No'}
• Home Internet Access    : {'Yes' if student_data.get('internet') == 'yes' else 'No'}
• Weekend Alcohol Intake  : {student_data.get('Walc', 1)} / 5

3. IDENTIFIED PROTECTIVE FACTORS & STRENGTHS
--------------------------------------------------------------------------------
{chr(10).join(['  • ' + s for s in recommendations.get('strengths', ['Regular course enrollment.'])])}

4. CRITICAL WARNING FLAGS & RISK FACTORS
--------------------------------------------------------------------------------
{chr(10).join(['  • ' + w for w in recommendations.get('warnings', ['No acute risk factors detected.'])])}

5. PRESCRIBED PEDAGOGICAL ACTION PLAN
--------------------------------------------------------------------------------
"""
    for item in recommendations.get('academic_actions', []):
        txt += f"  • [ACADEMIC - {item.get('priority', 'Medium')} Priority] {item.get('title', '')}: {item.get('detail', '')}\n"
    for item in recommendations.get('attendance_actions', []):
        txt += f"  • [ATTENDANCE - {item.get('priority', 'High')} Priority] {item.get('title', '')}: {item.get('detail', '')}\n"
    for item in recommendations.get('lifestyle_actions', []):
        txt += f"  • [LIFESTYLE - {item.get('priority', 'Medium')} Priority] {item.get('title', '')}: {item.get('detail', '')}\n"
    for item in recommendations.get('support_actions', []):
        txt += f"  • [SUPPORT - {item.get('priority', 'Medium')} Priority] {item.get('title', '')}: {item.get('detail', '')}\n"

    txt += f"""
6. RECOMMENDED PERSONALIZED WEEKLY TIMETABLE
--------------------------------------------------------------------------------
"""
    for day, desc in recommendations.get('weekly_schedule', {}).items():
        txt += f"  • {day.ljust(10)}: {desc}\n"

    txt += f"""
7. AI MODEL ACCURACY, TRANSPARENCY & LIMITATIONS
--------------------------------------------------------------------------------
• Stacking Classifier Accuracy : 92.41% (Evaluated on 79 unseen test students)
• Stacking Classifier Recall   : 92.31% (92.3% of struggling students caught)
• Stacking Classifier F1-Score : 0.889
• SGD Regression R² Score      : 0.805 (Over 80% of grade variance explained)
• SGD Regression RMSE          : 2.034 marks (On 0 to 20 Portuguese scale)

IMPORTANT LIMITATIONS:
Reported classification accuracy and regression R² reflect empirical benchmarks on
a test set of 79 students and do not represent a guarantee for future predictions.
Regression R² measures the continuous fit and should not be confused with discrete
pass/fail classification accuracy. This application functions as an early-warning
pedagogical decision support system and should not replace professional educators.
================================================================================
"""
    return txt
