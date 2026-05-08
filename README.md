# Clasificador de Lesiones Cutáneas

Esta es una aplicación web interactiva desarrollada con **Streamlit** que funciona como un Sistema de Soporte a la Decisión Clínica (DSS) para el pre-diagnóstico de lesiones cutáneas. 

El proyecto integra un modelo de **Machine Learning multimodal** que analiza simultáneamente la imagen dermatoscópica de la lesión y los metadatos clínicos del paciente (edad, sexo y localización anatómica) para ofrecer una predicción precisa.

---

## Características
- **Análisis Multimodal:** Fusión de características visuales y datos tabulares.
- **Reducción de Dimensionalidad:** Uso de Mutual Information y Análisis Discriminante Lineal (LDA).
- **Interfaz Intuitiva:** Desarrollada puramente en Python con Streamlit.
- **Resultados Claros:** Alertas visuales para lesiones malignas y desglose del nivel de confianza (probabilidades) de las enfermedades predichas.

---

## Contenido del Repositorio
Para que la aplicación funcione correctamente, este repositorio incluye:
- `app.py`: El código fuente principal de la aplicación web.
- `requirements.txt`: Lista de las librerías de Python necesarias para ejecutar el proyecto.
- **Archivos `.pkl`**: Componentes del pipeline de Machine Learning ya entrenados y listos para inferencia:
  - `modelo_logreg.pkl` (o similar): El modelo predictivo final.
  - `lda.pkl` y `top_mi_indices.pkl`: Modelos de extracción y selección de características.
  - `scaler_age.pkl`: Normalizador de la edad del paciente.
  - `label_encoder.pkl` y `columnas_meta.pkl`: Codificadores de las etiquetas y la estructura de los datos clínicos.

---

## Cómo ejecutar la aplicación en local

Si deseas probar la aplicación en tu propio ordenador, sigue estos sencillos pasos:

### 1. Requisitos previos
Asegúrate de tener instalado [Python](https://www.python.org/downloads/) en tu sistema.

### 2. Clonar o descargar el repositorio
Puedes clonar este repositorio usando Git o descargar los archivos directamente en formato `.zip` y descomprimirlos en una carpeta.
```bash
git clone [https://github.com/TU-USUARIO/TU-REPOSITORIO.git](https://github.com/TU-USUARIO/TU-REPOSITORIO.git)
cd TU-REPOSITORIO
