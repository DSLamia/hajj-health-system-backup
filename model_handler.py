import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def predict_logic(input_df, target_audience, has_chronic=False, disease_detail="none", diet_status="follows",
                  bed_capacity=None, occupied_beds=0):
    # استدعاء مكتبة ONNX
    import onnxruntime as ort

    # مسارات الملفات الديناميكية
    MODEL_PATH = os.path.join(BASE_DIR, 'ai_models', 'hajj_health_model.onnx')
    SCALER_PATH = os.path.join(BASE_DIR, 'ai_models', 'scaler.pkl')

    #  قراءة قيم الطقس
    try:
        temp_raw = input_df.iloc[0, 2]
        hum = input_df.iloc[0, 3]
    except Exception:
        temp_raw, hum = 35.0, 45.0  #قيم افتراضية طارئة فقط

    temp = (temp_raw * 0.98) + (temp_raw * (hum / 100) * 0.02)
    temp = round(temp + 0.3, 2)

    if bed_capacity is None:
        try:
            bed_capacity = input_df.iloc[0, 7]
        except Exception:
            bed_capacity = 10240

    # تشغيل التنبؤ عبر ONNX مع حماية السكالر من الحقول الناقصة
    try:
            scaler = joblib.load(SCALER_PATH)
            session = ort.InferenceSession(MODEL_PATH)

            raw_values = input_df.values

            scaled_input = scaler.transform(raw_values).astype(np.float32)

            input_name = session.get_inputs()[0].name
            prediction = session.run(None, {input_name: scaled_input})[0]

            heatstroke_count = int(max(0, prediction[0][0]))

    except Exception as e:
            print(f"🚨 ONNX/Scaler Critical Failure: {e}")
            raise e

    try:
        age_group_enc = input_df.iloc[0, 0]
    except Exception:
        age_group_enc = 1

    ratio = heatstroke_count / bed_capacity if bed_capacity > 0 else 0

    # 2. تصنيف مستوى الحرارة
    if temp >= 40:
        heat_level, color = "High", "red"
    elif 30 <= temp < 40:
        heat_level, color = "Moderate", "orange"
    else:
        heat_level, color = "Low", "green"

    # 3. منطق الحجاج (Pilgrim)
    if str(target_audience).lower() == "pilgrim":
        p_risk_points = 0

        disease_weights = {
            "heart": 4,
            "asthma": 3.5,
            "hypertension": 3,
            "neurological": 2.5,
            "diabetes1": 2,
            "diabetes2": 2,
            "cancer": 1.5,
            "hepatitis": 1,
            "rheumatism": 1,
            "none": 0
        }

        if has_chronic:
            p_risk_points += disease_weights.get(disease_detail, 1)

        if has_chronic and diet_status == "not_follows":
            p_risk_points += 0.25

        if age_group_enc >= 2:
            p_risk_points += 2

        if heat_level == "High" and p_risk_points >= 5:
            risk = "High"
            color = "red"
            rec = [
                f" 🚨  درجة الحرارة ({int(temp)}°C) مرتفعة جداً وتشكل خطورة على سلامتك.",
                "يرجى البقاء في مكان بارد وتجنب التحرك أو بذل أي مجهود بدني حالياً.",
                "احرص على شرب السوائل بانتظام لتعويض ما يفقده الجسم.",
                "نرجو منك التوجه لأقرب نقطة طبية فوراً في حال الشعور بأي إعياء."
            ]
        elif heat_level == "Moderate" and p_risk_points >= 3:
            risk = "Moderate"
            color = "orange"
            rec = [
                f"⚠️ الأجواء حالياً ({int(temp)}°C) تتطلب منك أخذ الحيطة والحذر.",
                "ننصحك باستخدام المظلة الشمسية عند الضرورة لتجنب الإجهاد الحراري.",
                "احرص على تناول السوائل والأملاح لتعويض المجهود البدني المبذول.",
                "يفضل تأجيل أي تحركات غير ضرورية حتى تنكسر حدة الشمس."
            ]
        else:
            risk = "Low"
            color = "green"
            rec = [
                f"✅ المؤشرات البيئية ({int(temp)}°C) ضمن النطاق الآمن والمستقر.",
                "يمكنك إكمال مناسكك مع الاستمرار في شرب السوائل كإجراء احترازي.",
                "حاول أخذ فترات راحة قصيرة بين الحين والآخر للحفاظ على نشاطك.",
                "تأكد من وجود تهوية جيدة في مكان إقامتك لضمان راحتك."
            ]

    # 4. منطق المسؤولين (Officer)
    else:
        try:
            actual_ratio = float(occupied_beds) / float(bed_capacity) if bed_capacity > 0 else 0
        except Exception:
            actual_ratio = 0

        occ_perc = int(actual_ratio * 100)

        if heat_level == "High" or actual_ratio >= 0.75 or ratio > 0.10:
            risk = "High"
            color = "red"
            rec = [
                f"🚨 تحذير حرج: نسبة الإشغال الميداني ({occ_perc}%) تجاوزت حد الأمان الحرج.",
                "مستويات الخطورة البيئية مرتفعة جداً؛ يرجى تفعيل خطة الطوارئ فوراً.",
                "توجيه مصفوفة الدعم الطبي الإضافي لتقليل الضغط على المستشفيات الحالية."
            ]
        elif actual_ratio >= 0.50:
            risk = "Moderate"
            color = "orange"
            rec = [
                f"⚠️ تنبيه متوسط: نسبة الإشغال الحالية ({occ_perc}%) في تصاعد مستمر.",
                "يرجى توجيه الحجاج للمسارات الأقل كثافة وإخطار المراكز الصحية الميدانية.",
                "رفع جاهزية الكوادر الطبية المتنقلة لاستقبال أي حالات إجهاد حراري محتملة."
            ]
        else:
            risk = "Low"
            color = "green"
            rec = [
                f"🟢 حالة المنظومة الطبية والبيئية مستقرة تماماً وجاهزيتها متميزة.",
                f"نسبة إشغال الأسرة الحالية هي {occ_perc}% وهي ضمن النطاق الطبيعي.",
                "توزيع الكثافات البشرية يسير بشكل ممتاز بالتنسيق مع غرف العمليات."
            ]

    # 5. تحديث وتوليد الرسم البياني للمشروع
    try:
        create_dashboard(heatstroke_count, bed_capacity, risk, color)
    except Exception as graph_err:
        print(f"Dashboard plot skipped: {graph_err}")

    return {
        "heatstroke": heatstroke_count,
        "risk_level": risk,
        "risk_color": color,
        "recommendation": rec,
        "graph_path": "static/report.png"
    }

def create_dashboard(heatstroke, beds, risk, color):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ['Predicted Heatstrokes', 'Available Beds']
    values = [heatstroke, beds]

    sns.barplot(x=labels, y=values, hue=labels, palette=[color, '#3498db'], ax=ax, legend=False)

    plt.title(f"Heatstroke Prediction Analysis (Status: {risk})", fontsize=14)
    plt.ylabel("Count")

    static_dir = os.path.join(BASE_DIR, 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, 'report.png'))
    plt.close(fig)