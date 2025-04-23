import streamlit as st
import pandas as pd
from modules.feature_extractor import prepare_input_features
from modules.predict import get_ensemble_prediction, success_category, percentile_rank


st.set_page_config(page_title="Course Success Predictor", layout="wide")
st.title("🎯 Course Success Score Predictor")


langs = ['en_US', 'en_GB', 'en_IN', 'ar_AR', 'ro_RO', 'zh_CN', 'tr_TR',
       'pt_BR', 'pl_PL', 'fr_CA', 'es_LA', 'ko_KR', 'zh_TW', 'es_ES',
       'sq_AL', 'hi_IN', 'th_TH', 'ja_JP', 'fr_FR', 'es_MX', 'es_CO',
       'ru_RU', 'de_DE', 'id_ID', 'ur_PK', 'pt_PT', 'it_IT', 'vi_VN',
       'az_AZ', 'fa_IR', 'te_IN', 'ml_IN', 'es_CL', 'cs_CZ', 'hu_HU',
       'es_VE', 'el_GR', 'ms_MY', 'uk_UA', 'he_IL', 'sw_KE', 'ta_IN',
       'zh_HK', 'bn_IN', 'nl_NL', 'ka_GE', 'my_MM']


st.subheader("Введите данные о курсе")

text_inputs = {
    'description': [
        st.text_input("Название курса"),
        st.text_input("Описание курса")
    ],
    'instructors' : [i.strip() for i in st.text_input(f"Инструкторы курса (описание работы, разделенное запятой)").strip(',').split(',')]
}

categorical_inputs = {
    'locale': st.selectbox("Язык курса (locale)", langs),
    'caption_locales': st.multiselect("Языки субтитров", options=langs, default=['en_US']),
    'is_paid': st.selectbox("Платный курс?", [True, False]),
    'has_closed_caption': st.selectbox("Есть субтитры?", [True, False]),
    'instructional_level_simple': st.selectbox("Уровень курса", ['All Levels', 'Beginner', 'Intermediate', 'Expert'])
}

numeric_inputs = {
    'num_published_lectures': st.number_input("Количество лекций", 0, 500, 10),
    'num_published_practice_tests': st.number_input("Количество тестов", 0, 100, 5),
    'content_length_practice_test_questions': st.number_input("Количество вопросов в тестах", 0, 1000, 100),
    'content_hours': st.number_input("Часы контента", 0.0, 100.0, 5.0),
}


if st.button("Предсказать успех"):    
    input_df = pd.DataFrame([numeric_inputs | categorical_inputs | text_inputs])
    features = prepare_input_features(input_df)
    
    prediction = get_ensemble_prediction(features)
    category = success_category(prediction)
    rank_percent = percentile_rank(prediction)
    
    st.success(f"✨ Прогнозируемый успех курса: {prediction:.2f}")
    st.success(f"✨ Предполагаемый уровень успешности: {category}")
    st.success(f"🏅 Курс входит в **топ-{100 - rank_percent:.1f}%** лучших курсов по успешности.")
    