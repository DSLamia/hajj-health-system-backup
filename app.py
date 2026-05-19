from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
import requests
from supabase import create_client, Client
from model_handler import predict_logic
import os

app = Flask(__name__)
CORS(app)

# إعدادات سوبابيس
SUPABASE_URL = "https://rmpmbnmmgxsbxcvcbkwb.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJtcG1ibm1tZ3hzYnhjdmNia3diIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE3ODcwMzgsImV4cCI6MjA4NzM2MzAzOH0.Piu2jTOwdfihFgEsELJyTHXChGgV95abKAy4-9lsAHc"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

WEATHER_API_KEY = "0aac0c7a97816848748a258ddcb625b0"
FEATURES = [
    'age_group', 'total_pilgrims', 'temperature', 'humidity', 'wind_speed',
    'hospitals_count', 'health_centers_count', 'total_beds', 'staff_count',
    'ambulances', 'chronic_input'
]


def get_makkah_weather():
    url = f"https://api.openweathermap.org/data/2.5/weather?q=Mecca,SA&appid={WEATHER_API_KEY}&units=metric"
    try:
        res = requests.get(url, timeout=5).json()
        if res.get("cod") == 200:
            return res['main']['temp'], res['main']['humidity'], res['wind']['speed'] * 3.6
    except Exception as e:
        print(f"Log: External Weather API failed, using defaults. Info: {e}")
    return 22.6, 45.0, 10.0


#  ==================== مسارات واجهات المستخدم (HTML Routing) ====================

@app.route('/')
def index():
    # الصفحة الرئيسية الأولى التي يراها الزائر (بوابة الحج والتلبية التفاعلية)
    return render_template('Untitled 4.html')


@app.route('/public-weather')
def public_weather():
    # لوحة طقس مكة المكرمة العامة والتوصيات الذكية للأفراد
    return render_template('pub.html')


@app.route('/employee-dashboard')
def employee_dashboard():
    # لوحة تحكم الموظفين والمسؤولين لمتابعة الإشغالات وإرسال التنبيهات
    return render_template('employee_dashboard.html')


@app.route('/pilgrim-dashboard')
def pilgrim_dashboard():
    # لوحة تحكم وتنبؤات الحجاج الشخصية
    return render_template('pilgrim_dashboard.html')


@app.route('/emergency')
def emergency():
    # صفحة الطوارئ والإسعاف وفرق الاستجابة العاجلة
    return render_template('emrg.html')


#  ==================== مسارات المعالجة والـ APIs الخلفية ====================

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}

        # 1. قراءة البيانات الأساسية القادمة من الواجهة
        temp = float(data.get('temperature', 32.2))
        humidity = float(data.get('humidity', 9.0))
        wind_speed = float(data.get('wind_speed', 10.0))
        crowd_density = float(data.get('crowding_density', 1.0))

        # قراءة بيانات السعة للمسؤولين
        bed_capacity = float(data.get('bed_capacity', 150.0))
        occupied_beds = float(data.get('occupied_beds', 45.0))

        # تحديد (pilgrim أو officer)
        target_audience = data.get('target_audience', 'officer')
        phone_number = data.get('phone_number')
        user_id = data.get('user_id')

        # قيم افتراضية للمسؤول
        age_group_enc = 1.0
        chronic_disease = 0.0
        has_chronic = False
        disease_detail = "none"
        diet_status = "follows"

        # فحص صيغة الـ UUID للمستخدم
        is_valid_uuid = False
        if user_id and len(str(user_id)) == 36 and '-' in str(user_id):
            is_valid_uuid = True

        # جلب بيانات الحاج الحقيقية من Supabase
        if phone_number or is_valid_uuid:
            try:
                user_query = supabase.table('profiles').select('age, chronic_diseases, disease_type, diet_compliance')
                if phone_number:
                    user_query = user_query.eq('phone_number', str(phone_number))
                elif is_valid_uuid:
                    user_query = user_query.eq('pilgrim_id', str(user_id))

                user_res = user_query.execute()

                if user_res.data and len(user_res.data) > 0:
                    profile = user_res.data[0]
                    raw_age = int(profile.get('age', 35))
                    age_group_enc = 0.0 if raw_age <= 15 else (2.0 if raw_age >= 61 else 1.0)

                    has_chronic = profile.get('chronic_diseases', False)
                    chronic_disease = 100.0 if has_chronic else 0.0
                    disease_detail = str(profile.get('disease_type', 'none')).lower()
                    diet_status = str(profile.get('diet_compliance', 'follows')).lower()
            except Exception as sb_e:
                print(f"Supabase context fetch skipped safely: {sb_e}")

        # 2. بناء Matrix الـ 11 واستدعاء المودل
        features_dict = {
            'Age_Group': [float(age_group_enc)],
            'Crowd_Density': [float(crowd_density)],
            'Temperature': [float(temp)],
            'Humidity': [float(humidity)],
            'Wind_Speed': [float(wind_speed)],
            'Hospitals_Count': [3.0],
            'Health_Centers_Count': [10.0],
            'Total_Bed_Capacity': [bed_capacity],
            'Staff_Count': [45.0],
            'Ambulance_Fleet_Size': [12.0],
            'Chronic_Disease_Input': [float(chronic_disease)]
        }

        import pandas as pd
        input_df = pd.DataFrame(features_dict)

        from model_handler import predict_logic
        result_model = predict_logic(input_df, temp)

        if isinstance(result_model, dict):
            heatstroke_count = int(result_model.get('heatstroke_predictions', result_model.get('prediction', 0)))
        else:
            heatstroke_count = int(result_model)

        ratio = heatstroke_count / bed_capacity if bed_capacity > 0 else 0

        # [منطق تصنيف مستوى الحرارة]
        if temp >= 40:
            heat_level, color = "High", "red"
        elif 30 <= temp < 40:
            heat_level, color = "Moderate", "orange"
        else:
            heat_level, color = "Low", "green"

        # [منطق الحجاج - Pilgrim]
        if str(target_audience).lower() == "pilgrim":
            p_risk_points = 0
            disease_weights = {
                "heart": 4, "asthma": 3.5, "hypertension": 3, "neurological": 2.5,
                "diabetes1": 2, "diabetes2": 2, "cancer": 1.5, "hepatitis": 1,
                "rheumatism": 1, "none": 0
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
                    f" 🚨 درجة الحرارة ({int(temp)}°C) مرتفعة جداً وتشكل خطورة على سلامتك.",
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

        # [منطق المسؤولين - Officer]
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

        #
        return jsonify({
            "status": "success",
            "heatstroke_predictions": heatstroke_count,
            "risk_level": risk,
            "risk_color": color,
            "recommendations": rec
        })

    except Exception as main_e:
        return jsonify({"status": "error", "message": f"Inference Error: {str(main_e)}"}), 400


@app.route('/api/send-report', methods=['POST'])
def send_report():
    try:
        data = request.json or {}
        report_data = {
            "location_name": data.get('location'),
            "readiness_level": data.get('type', 'General'),
            "status": "Active"
        }
        supabase.from_("emergency_team").insert(report_data).execute()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == '__main__':
    # تهيئة رقم المنفذ المتوافق مع خوادم الاستضافة السحابية
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)