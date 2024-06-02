from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from statistics import mode

mapping = {'nothing': 0,
 'Common Cold': 1,
 'Eczema': 2,
 'Hyperthyroidism': 3,
 'Allergic Rhinitis': 4,
 'Anxiety Disorders': 5,
 'Diabetes': 6,
 'Gastroenteritis': 7,
 'Pancreatitis': 8,
 'Rheumatoid Arthritis': 9,
 'Dengue Fever': 10,
 'Hepatitis': 11,
 'Kidney Cancer': 12,
 'Migraine': 13,
 'Muscular Dystrophy': 14,
 'Sinusitis': 15,
 'Ulcerative Colitis': 16,
 'Asthma': 17,
 'Osteoporosis': 18,
 'Atherosclerosis': 19,
 'Chronic Obstructive Pulmonary': 20,
 'Epilepsy': 21,
 'Hypertension': 22,
 'Obsessive-Compulsive Disorde': 23,
 'Pneumonia': 24,
 'Psoriasis': 25,
 'Rubella': 26,
 'Urinary Tract Infection (UTI)': 27,
 'Depression': 28,
 'Influenza': 29,
 'Kidney Disease': 30,
 'Liver Cancer': 31,
 'Liver Disease': 32,
 'Stroke': 33,
 'Acne': 34,
 'Brain Tumor': 35,
 'Bronchitis': 36,
 'Cystic Fibrosis': 37,
 'Glaucoma': 38,
 'Osteoarthritis': 39,
 'Rabies': 40,
 'Lung Cancer': 41,
 'Urinary Tract Infection': 42,
 'Autism Spectrum Disorder (ASD)': 43,
 "Crohn's Disease": 44,
 'Hyperglycemia': 45,
 'Melanoma': 46,
 'Ovarian Cancer': 47,
 'Turner Syndrome': 48,
 'Zika Virus': 49,
 'Hypothyroidism': 50,
 'Anemia': 51,
 'Cholera': 52,
 'Endometriosis': 53,
 'Sepsis': 54,
 'Sleep Apnea': 55,
 'Multiple Sclerosis': 56,
 'Appendicitis': 57,
 'Esophageal Cancer': 58,
 'HIV/AIDS': 59,
 'Marfan Syndrome': 60,
 "Parkinson's Disease": 61,
 'Breast Cancer': 62,
 'Coronary Artery Disease': 63,
 'Measles': 64,
 'Osteomyelitis': 65,
 'Polio': 66,
 'Bladder Cancer': 67,
 'Otitis Media (Ear Infection)': 68,
 'Tourette Syndrome': 69,
 "Alzheimer's Disease": 70,
 'Cholecystitis': 71,
 'Chronic Obstructive Pulmonary Disease (COPD)': 72,
 'Prostate Cancer': 73,
 'Schizophrenia': 74}

def use_multi_model(Fever,Cough,Fatigue,Difficulty_Breathing,Age,Gender,Blood_Pressure,Cholesterol_Level):
    usercase = [Fever,Cough,Fatigue,Difficulty_Breathing,Age,Gender]
    X_resampled_df = pd.read_csv(r"D:\Code-Projects\University\Spring2024\Desision Support\Medical-Consultant-Backend\src\data\X.csv")
    y_resampled_df = pd.read_csv(r"D:\Code-Projects\University\Spring2024\Desision Support\Medical-Consultant-Backend\src\data\y.csv")['Disease']
    knn_loaded = joblib.load(r'D:\Code-Projects\University\Spring2024\Desision Support\Medical-Consultant-Backend\src\data\knn_model.pkl')
    nb_loaded = joblib.load(r'D:\Code-Projects\University\Spring2024\Desision Support\Medical-Consultant-Backend\src\data\nb_model.pkl')
    svm_loaded = joblib.load(r'D:\Code-Projects\University\Spring2024\Desision Support\Medical-Consultant-Backend\src\data\svm_model.pkl')

    if Blood_Pressure == "High":
        usercase.extend([1,0,0])
    elif Blood_Pressure == "Low":
        usercase.extend([0,1,0])
    else:
        usercase.extend([0,0,1])

    if Cholesterol_Level == "High":
        usercase.extend([1,0,0])
    elif Cholesterol_Level == "Low":
        usercase.extend([0,1,0])
    else:
        usercase.extend([0,0,1])   


    case_df = pd.DataFrame([usercase])
    
    similarities = cosine_similarity(X_resampled_df, case_df)
    
    nearest_index = np.argmax(similarities)

    simularity_answer = y_resampled_df[nearest_index] 
    knn_answer = knn_loaded.predict(case_df)[0]
    nb_answer =  nb_loaded.predict(case_df)[0]
    svm_answer = svm_loaded.predict(case_df)[0]

    def custom_mode(arr):
        try:
            mode_value = mode(arr)
        except:  
            mode_value = max(arr)
        return mode_value
    answer = [k for k,v in mapping.items() if v == custom_mode([simularity_answer,knn_answer,nb_answer,svm_answer])][0]
    return answer

@csrf_exempt
def predict(request):
    data = json.loads(request.body)
    answer = use_multi_model(*data['case'])
    return JsonResponse({"answer":answer})
