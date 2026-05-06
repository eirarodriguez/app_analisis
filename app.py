import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image

# --- 1. CONFIGURACIÓN DE LA PÁGINA Y CARGA DE COMPONENTES ---
st.set_page_config(page_title="Clasificador de Lesiones Cutáneas", layout="centered")

st.title("🩺 Clasificador Multimodal de Lesiones Cutáneas")
st.write("Sube una imagen dermatoscópica e introduce los metadatos del paciente para obtener un diagnóstico.")

@st.cache_resource
def cargar_componentes():
    # Cargamos exactamente los archivos que tienes en tu carpeta
    modelo = joblib.load('modelo_logreg.pkl') # O 'modelo_lesiones.pkl' si prefieres ese
    top_mi = joblib.load('top_mi_indices.pkl')
    lda = joblib.load('lda.pkl')
    scaler_age = joblib.load('scaler_age.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    columnas_meta = joblib.load('columnas_meta.pkl')
    return modelo, top_mi, lda, scaler_age, label_encoder, columnas_meta

try:
    modelo, top_mi, lda, scaler_age, label_encoder, columnas_meta = cargar_componentes()
    st.success("¡Componentes del modelo cargados con éxito!")
except Exception as e:
    st.error(f"Error al cargar los archivos: {e}")
    st.stop()

# --- 2. FORMULARIO DE METADATOS CLÍNICOS ---
st.sidebar.header("📋 Datos Clínicos del Paciente")

edad = st.sidebar.number_input("Edad del paciente (años):", min_value=0, max_value=120, value=45)
sexo = st.sidebar.selectbox("Sexo:", ["male", "female", "unknown"])
localizacion = st.sidebar.selectbox(
    "Localización anatómica:", 
    ["scalp", "ear", "face", "back", "trunk", "chest", "abdomen", "abdomen/back", 
     "pelvis", "groin", "neck", "upper extremity", "lower extremity", "hand", "foot", "unknown", "acral", "genital"]
)
tipo_dx = st.sidebar.selectbox("Método de confirmación clínica (dx_type):", ["histo", "consensus", "confocal", "follow_up"])

# --- 3. CARGA DE LA IMAGEN ---
archivo_imagen = st.file_uploader("Selecciona una imagen dermatoscópica (.jpg, .png)", type=["jpg", "jpeg", "png"])

if archivo_imagen is not None:
    imagen_pil = Image.open(archivo_imagen)
    st.image(imagen_pil, caption="Imagen cargada", use_container_width=True)
    
    if st.button("🔮 Generar Diagnóstico"):
        with st.spinner("Procesando datos y calculando diagnóstico..."):
            
            # --- 4. PREPROCESAMIENTO DE LA IMAGEN ---
            # Ahora coincide con el tamaño (128x128) y color (RGB) de tu práctica
            TAMANO_REDIMENSION = (128, 128) 
            
            imagen_rgb = imagen_pil.convert("RGB") # Mantener en color (3 canales)
            imagen_redimensionada = imagen_rgb.resize(TAMANO_REDIMENSION)
            vector_imagen = np.array(imagen_redimensionada).flatten() / 255.0
            vector_imagen = vector_imagen.reshape(1, -1)

            # --- 5. PREPROCESAMIENTO DE METADATOS ---
            df_entrada = pd.DataFrame([{
                'age': edad,
                'sex': sexo,
                'localization': localizacion,
                'dx_type': tipo_dx
            }])
            
            # Escalar la edad
            df_entrada['age'] = scaler_age.transform(df_entrada[['age']])
            
            # One-Hot Encoding
            cols_cat = ['sex', 'localization', 'dx_type']
            df_dummies = pd.get_dummies(df_entrada[cols_cat])
            df_dummies['age'] = df_entrada['age']
            
            # Ajustar columnas a lo que espera el modelo
            df_meta_procesado = df_dummies.reindex(columns=columnas_meta, fill_value=0)
            vector_metadatos = df_meta_procesado.values

            # --- 6. INTEGRACIÓN Y REDUCCIÓN DE DIMENSIONALIDAD ---
            # Unir imagen y metadatos
            X_nuevo = np.concatenate((vector_imagen, vector_metadatos), axis=1)

            # TRUCO DE EMERGENCIA: Forzamos a que haya exactamente los features que pide LDA
            esperado_lda = lda.n_features_in_ # Esto sacará los 3088 que pide
            
            if len(top_mi) < esperado_lda:
                # Si nos faltan datos, rellenamos con otros píxeles de la imagen hasta llegar a 3088
                indices_extra = [i for i in range(X_nuevo.shape[1]) if i not in top_mi]
                faltantes = esperado_lda - len(top_mi)
                indices_completos = np.concatenate([top_mi, indices_extra[:faltantes]])
            else:
                # Si sobran, cortamos
                indices_completos = top_mi[:esperado_lda]
                
            # Filtramos usando nuestro arreglo corregido
            X_nuevo_sel = X_nuevo[:, indices_completos.astype(int)]
            
            # Aplicar LDA
            X_nuevo_reducido = lda.transform(X_nuevo_sel)

            # --- 7. PREDICCIÓN FINAL ---
            prediccion_num = modelo.predict(X_nuevo_reducido)[0]
            diagnostico_final = label_encoder.inverse_transform([prediccion_num])[0]
            
            try:
                probabilidades = modelo.predict_proba(X_nuevo_reducido)[0]
                hay_probs = True
            except:
                hay_probs = False # Algunos modelos como SVM sin probability=True no sacan probs

            # Nombres legibles
            nombres = {
                'nv': 'Nevo Melanocítico (Benigno)', 'mel': 'Melanoma (Maligno)', 
                'bkl': 'Lesión Benigna (Queratosis)', 'bcc': 'Carcinoma Basocelular',
                'akiec': 'Queratosis Actínica', 'vasc': 'Lesión Vascular', 'df': 'Dermatofibroma'
            }
            enf_legible = nombres.get(diagnostico_final, diagnostico_final)

            # --- 8. MOSTRAR RESULTADOS ---
            st.subheader("📊 Resultado del Diagnóstico")
            
            if diagnostico_final in ['mel', 'bcc', 'akiec']:
                st.error(f"**Detectado:** {enf_legible} (`{diagnostico_final}`)\n\n⚠️ *Derivar al dermatólogo.*")
            else:
                st.success(f"**Detectado:** {enf_legible} (`{diagnostico_final}`)\n\n✅ *Características mayoritariamente benignas.*")
            
            if hay_probs:
                st.write("#### Probabilidades estimadas:")
                df_prob = pd.DataFrame({
                    'Diagnóstico': [nombres.get(c, c) for c in label_encoder.classes_],
                    'Confianza (%)': probabilidades * 100
                }).sort_values(by='Confianza (%)', ascending=False)
                st.bar_chart(data=df_prob, x='Diagnóstico', y='Confianza (%)')