import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image

#------------------------------
# CONFIGURACIÓN DE LA PÁGINA
#------------------------------
st.set_page_config(page_title="Clasificador de Lesiones Cutáneas", layout="centered")

st.title("Clasificador de Lesiones Cutáneas")
st.write("Sube una imagen dermatoscópica e introduce los metadatos del paciente para obtener un diagnóstico.")

@st.cache_resource
def cargar_componentes():
    modelo = joblib.load('modelo_logreg.pkl') 
    top_mi = joblib.load('top_mi_indices.pkl')
    lda = joblib.load('lda.pkl')
    scaler_age = joblib.load('scaler_age.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    columnas_meta = joblib.load('columnas_meta.pkl')
    return modelo, top_mi, lda, scaler_age, label_encoder, columnas_meta

try:
    modelo, top_mi, lda, scaler_age, label_encoder, columnas_meta = cargar_componentes()
except Exception as e:
    st.error(f"Error al cargar los archivos: {e}")
    st.stop()

#------------------------------
# FORMULARIO DE DATOS
#------------------------------
st.sidebar.header("Datos Clínicos")

edad = st.sidebar.number_input("Edad del paciente (años):", min_value=0, max_value=120, value=45)
sexo = st.sidebar.selectbox("Sexo:", ["male", "female", "unknown"])
localizacion = st.sidebar.selectbox(
    "Localización anatómica:", 
    ["scalp", "ear", "face", "back", "trunk", "chest", "abdomen", "abdomen/back", 
     "pelvis", "groin", "neck", "upper extremity", "lower extremity", "hand", "foot", "unknown", "acral", "genital"]
)
tipo_dx = st.sidebar.selectbox("Método de confirmación clínica (dx_type):", ["histo", "consensus", "confocal", "follow_up"])

# ------------------------------
# CARGA DE IMAGEN Y PREDICCIÓN
#------------------------------
archivo_imagen = st.file_uploader("Selecciona una imagen dermatoscópica (.jpg, .png)", type=["jpg", "jpeg", "png"])

if archivo_imagen is not None:
    imagen_pil = Image.open(archivo_imagen)
    st.image(imagen_pil, caption="Imagen cargada", use_container_width=True)
    
    if st.button("GENERAR DIAGNÓSTICO"):
        with st.spinner("Procesando datos y calculando diagnóstico..."):
            
            # preprocesamiento de la imagen
            TAMANO_REDIMENSION = (128, 128) 
            imagen_rgb = imagen_pil.convert("RGB") 
            imagen_redimensionada = imagen_rgb.resize(TAMANO_REDIMENSION)
            vector_imagen = np.array(imagen_redimensionada).flatten() / 255.0
            vector_imagen = vector_imagen.reshape(1, -1)

            # preprocesamiento de metadatos
            df_entrada = pd.DataFrame([{
                'age': edad,
                'sex': sexo,
                'localization': localizacion,
                'dx_type': tipo_dx
            }])
            
            # escalado de edad
            df_entrada['age'] = scaler_age.transform(df_entrada[['age']])
            
            # One-Hot Encoding
            cols_cat = ['sex', 'localization', 'dx_type']
            df_dummies = pd.get_dummies(df_entrada[cols_cat])
            df_dummies['age'] = df_entrada['age']
            
            # ajustar columnas
            df_meta_procesado = df_dummies.reindex(columns=columnas_meta, fill_value=0)
            vector_metadatos = df_meta_procesado.values

            # ------------------------------
            # PREDICCIÓN FINAL
            #------------------------------

            # unir imagen y metadatos
            X_nuevo = np.concatenate((vector_imagen, vector_metadatos), axis=1)

            # selección de características (MI)
            X_nuevo_sel = X_nuevo[:, top_mi]
            
            # extracción de características (LDA)
            X_nuevo_reducido = lda.transform(X_nuevo_sel)

            # prediccion final
            prediccion_num = modelo.predict(X_nuevo_reducido)[0]
            diagnostico_final = label_encoder.inverse_transform([prediccion_num])[0]
            
            try:
                probabilidades = modelo.predict_proba(X_nuevo_reducido)[0]
                hay_probs = True
            except:
                hay_probs = False # 

            nombres = {
                'nv': 'Nevo Melanocítico (Benigno)', 'mel': 'Melanoma (Maligno)', 
                'bkl': 'Lesión Benigna (Queratosis)', 'bcc': 'Carcinoma Basocelular',
                'akiec': 'Queratosis Actínica', 'vasc': 'Lesión Vascular', 'df': 'Dermatofibroma'
            }
            enf_legible = nombres.get(diagnostico_final, diagnostico_final)

            # ---------------------------
            # RESULTADOS Y VISUALIZACIÓN
            # ---------------------------
            st.markdown("---") 
            st.subheader("Resultado del Diagnóstico")
            
            if diagnostico_final in ['mel', 'bcc', 'akiec']:
                st.error(f"**Detección Principal:** {enf_legible} (`{diagnostico_final}`)")
                st.warning("*Se recomienda encarecidamente derivar al dermatólogo para revisión.*")
            else:
                st.success(f"**Detección Principal:** {enf_legible} (`{diagnostico_final}`)")
                st.info("*El análisis sugiere características mayoritariamente benignas.*")
            
            if hay_probs:
                st.markdown("Nivel de Confianza ")
                
                df_prob = pd.DataFrame({
                    'Diagnóstico': [nombres.get(c, c) for c in label_encoder.classes_],
                    'Confianza': probabilidades
                }).sort_values(by='Confianza', ascending=False)
                
                # Mostrar las 3 probabilidades más altas 
                for index, row in df_prob.head(3).iterrows():
                    col_texto, col_porcentaje = st.columns([3, 1])
                    with col_texto:
                        st.write(f"**{row['Diagnóstico']}**")
                        st.progress(float(row['Confianza'])) # Barra de progreso
                    with col_porcentaje:
                        st.write(f"{row['Confianza'] * 100:.2f}%")
                
                # Desplegable para ver todas las probabilidades
                with st.expander("Ver todas las probabilidades calculadas"):
                    # Tabla 
                    st.dataframe(
                        df_prob.style.format({'Confianza': '{:.2%}'}).background_gradient(cmap='Oranges'),
                        use_container_width=True,
                        hide_index=True
                    )