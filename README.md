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
  - `modelo_logreg.pkl`: El modelo predictivo final.
  - `lda.pkl` y `top_mi_indices.pkl`: Modelos de extracción y selección de características.
  - `scaler_age.pkl`: Normalizador de la edad del paciente.
  - `label_encoder.pkl` y `columnas_meta.pkl`: Codificadores de las etiquetas y la estructura de los datos clínicos.

---
## Acceso Directo (Versión en la Nube)
La aplicación se encuentra desplegada y disponible públicamente en:
https://lesionescutaneas.streamlit.app/

## Ejecución de la aplicación en local
Si deseas probar la aplicación en tu propio ordenador, sigue estos sencillos pasos:

### 1. Requisitos previos
Asegúrate de tener instalado **Python 3.9 o superior**. Puedes descargarlo en [python.org](https://www.python.org/).

### 2. Preparar los archivos
Descarga este repositorio y abre una terminal (o PowerShell / Anaconda Prompt) en la carpeta donde se encuentren los archivos.

### 3. Crear y activar un entorno virtual (Recomendado)
Esto crea un espacio aislado para que las librerías no choquen con otros proyectos:

**En Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```
**En macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```
### 4. Instalación completa de dependencias
Con el entorno activado, instala todas las librerías necesarias ejecutando:
```bash
pip install -r requirements.txt
```
### 5. Lanzar la aplicación
Una vez finalizada la instalación, ejecuta el siguiente comando:
```bash
streamlit run app.py
```
###⚠️ Aviso Legal / Disclaimer
Esta aplicación ha sido desarrollada con fines académicos dentro del marco del Máster en Ingeniería Biomédica. Las predicciones tienen carácter orientativo y estadístico basado en los datos de entrenamiento. Esta herramienta no sustituye bajo ningún concepto el diagnóstico de un dermatólogo profesional. Ante cualquier duda sobre una lesión cutánea, consulte siempre con un especialista.
