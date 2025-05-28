import streamlit as st
import numpy as np
import plotly.express as px
from PIL import Image
import requests
from io import BytesIO # Para manejar la descarga de la imagen clasificada
import time # Para el spinner

# --- Configuración de página ---
st.set_page_config(page_title="Clasificador de Residuos Inteligente", layout="wide", initial_sidebar_state="expanded")

# --- Cargar modelo ---
@st.cache_resource
def cargar_modelo():
    with st.spinner("Cargando modelo de IA... Esto puede tardar un momento."):
        # Asegúrate de que "modelo_residuos.keras" esté en el mismo directorio o especifica la ruta completa
        try:
            model = tf.keras.models.load_model("modelo_residuos.keras")
            return model
        except Exception as e:
            st.error(f"Error al cargar el modelo: {e}. Asegúrate de que 'modelo_residuos.keras' esté en la misma carpeta.")
            st.stop() # Detiene la ejecución de la app si el modelo no carga

modelo = cargar_modelo()
clases_residuos = ['cartón', 'vidrio', 'metal', 'papel', 'plástico', 'basura']

# --- Categorías extendidas ---
tipo_residuo = {
    'cartón': 'Reciclable',
    'vidrio': 'Reciclable',
    'metal': 'Reciclable',
    'papel': 'Reciclable',
    'plástico': 'Reciclable',
    'basura': 'Inorgánico'
}

# --- Información detallada por clase ---
info_detalle_clase = {
    'cartón': "El **cartón** debe estar limpio y seco. Dóblalo para ahorrar espacio y quita cualquier residuo de comida o cinta adhesiva excesiva. ¡Las cajas de pizza grasosas, los vasos de café y cartones de leche/jugo plastificados **no van al reciclaje**!",
    'vidrio': "Las **botellas y frascos de vidrio** (transparente, verde, ámbar) son reciclables. Lávalos bien y quita tapas y etiquetas si es posible. El vidrio roto, espejos, bombillas, cerámica y vasos de cristal **no se reciclan** en el contenedor de vidrio debido a riesgos de seguridad y diferente composición.",
    'metal': "Las **latas de aluminio y acero**, así como envases de alimentos (limpios) y aerosoles vacíos (sin contenido peligroso) son reciclables. Asegúrate de que no contengan líquidos. Los metales grandes o complejos requieren centros de acopio específicos.",
    'papel': "El **papel blanco o de oficina**, periódicos, revistas, folletos son reciclables. Debe estar limpio y seco. Evita reciclar papel encerado, papel de calcar, papel de fotos, papel carbón o papel con adhesivos excesivos. Las servilletas y toallas de papel usadas **tampoco son reciclables**.",
    'plástico': "Revisa el símbolo de reciclaje (triángulo con un número). Los **plásticos PET (1) y HDPE (2)** son los más comunes y generalmente aceptados. Límpialos y aplástalos si puedes para ahorrar volumen. Otros plásticos (como PVC-3, LDPE-4, PP-5, PS-6 y Otros-7) a menudo **no son reciclables** en los sistemas municipales estándar.",
    'basura': "Esta categoría incluye **residuos inorgánicos no reciclables** como papel sucio, toallas sanitarias, pañales, cerámica rota, telgopor (EPS), colillas de cigarro, chicles, etc. Son residuos que no tienen un uso posterior en los ciclos de reciclaje actuales y deben ir al vertedero o incineración."
}

# --- Estilos personalizados ---
st.markdown(
    """
    <style>
    .main {
        background-color: #f4f6f8; /* Un color neutro más claro */
    }
    .stButton > button {
        color: white;
        background-color: #28a745; /* Verde más vibrante */
        font-weight: bold;
        border-radius: 8px; /* Bordes redondeados */
        padding: 0.75rem 1.5rem;
        border: none;
        transition: background-color 0.3s ease; /* Transición suave para hover */
    }
    .stButton > button:hover {
        background-color: #218838; /* Oscurece al pasar el ratón */
        transform: translateY(-2px); /* Pequeño efecto de elevación */
        box-shadow: 0 4px 8px rgba(0,0,0,0.2); /* Sombra al pasar el ratón */
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #1a4d2e; /* Verde oscuro */
    }
    .stMarkdown h4 {
        color: #388e3c; /* Verde medio */
    }
    .result-box {
        padding: 25px;
        background-color: #e0f2f7; /* Azul claro para resultados */
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
        animation: fadeIn 1s ease-out; /* Animación de aparición */
    }
    .result-box h2 {
        color: #0d47a1; /* Azul oscuro */
        font-size: 2.2em; /* Tamaño de fuente para la clase */
    }
    .sidebar .sidebar-content {
        background-color: #dcedc8; /* Un verde claro para la sidebar */
        padding-top: 20px;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .info-card {
        background-color: #f0f7f4;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .info-card.red {
        border-left: 5px solid #dc3545; /* Rojo para no reciclable */
    }
    .info-card.yellow {
        border-left: 5px solid #ffc107; /* Amarillo para orgánicos */
    }
    
    /* Nuevos estilos añadidos */
    
    /* Mejoras para las pestañas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 20px;
        border-radius: 8px 8px 0 0;
        transition: all 0.3s ease;
        background-color: #f0f0f0;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e0e0e0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #28a745;
        color: white !important;
        font-weight: bold;
    }
    
    /* Mejoras para los expanders */
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .stExpander summary {
        font-weight: bold;
        padding: 12px;
        background-color: #f8f9fa;
        border-radius: 8px;
    }
    .stExpander summary:hover {
        background-color: #e9ecef;
    }
    .stExpander[open] summary {
        background-color: #28a745;
        color: white;
    }
    
    /* Mejoras para los inputs */
    .stFileUploader {
        border: 2px dashed #28a745;
        border-radius: 12px;
        padding: 20px;
        background-color: rgba(40, 167, 69, 0.05);
    }
    .stFileUploader:hover {
        background-color: rgba(40, 167, 69, 0.1);
    }
    
    /* Efectos hover para las tarjetas */
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    /* Animación para los botones */
    .stButton > button {
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.03); }
        100% { transform: scale(1); }
    }
    
    /* Mejoras para las métricas */
    [data-testid="stMetric"] {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #6c757d;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #28a745;
    }
    
    /* Efecto de gradiente para el título principal */
    .stMarkdown h1 {
        background: linear-gradient(45deg, #28a745, #0d47a1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
    }
    
    /* Sombras mejoradas para las imágenes */
    .stImage img {
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stImage img:hover {
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }
    
    /* Mejoras para los gráficos */
    .stPlotlyChart {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Efecto de onda para el botón principal */
    @keyframes wave {
        0% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0.7); }
        100% { box-shadow: 0 0 0 20px rgba(40, 167, 69, 0); }
    }
    .stButton > button[data-testid="baseButton-primary"] {
        animation: wave 2s infinite;
    }
    
    /* Mejoras para el modo oscuro */
    .main .block-container.dark {
        background-color: #1a1a1a;
    }
    .dark .sidebar .sidebar-content {
        background-color: #2a2a2a;
    }
    .dark .info-card {
        background-color: #2a2a2a;
        color: white;
    }
    
    /* Responsividad mejorada */
    @media (max-width: 768px) {
        .stMarkdown h1 {
            font-size: 1.8rem;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 12px;
            font-size: 0.9rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Historial de clasificaciones ---
if "historial" not in st.session_state:
    st.session_state["historial"] = []

# --- Opciones de usuario (Tema) ---
tema = st.sidebar.selectbox("🌈 Elige un tema", ["Estándar", "Modo oscuro", "Ecológico"])
if tema == "Modo oscuro":
    st.markdown('<style>body{background-color: #2E3B55; color: white;}</style>', unsafe_allow_html=True)
elif tema == "Ecológico":
     pass


# --- Pestañas ---
pestana_clasificador, pestana_info, pestana_historial, pestana_faq, pestana_about = st.tabs(["🧠 Clasificador", "📘 Información Educativa", "📜 Historial", "❓ Preguntas Frecuentes", "ℹ️ Acerca de"])


with pestana_clasificador:
    st.title("🌍 Clasificador Inteligente de Residuos")
    st.markdown(
        """
        Este sistema utiliza **Inteligencia Artificial** para identificar residuos y clasificarlos como **reciclables** o **inorgánicos**.
        Sube una imagen y descubre a qué categoría pertenece tu residuo: **cartón, vidrio, metal, papel, plástico o basura**.
        """
    )

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/927/927567.png", width=120)
        st.header("📈 Métricas del Modelo")
        st.metric("Precisión (accuracy)", "0.73")
        st.metric("Macro Promedio", "0.70")
        st.metric("Promedio Ponderado", "0.73")
        st.info("Estas métricas indican qué tan bien el modelo predice la clase correcta en general y su rendimiento en diferentes tipos de residuos.")
        st.markdown("---")
        st.subheader("⚙️ Opciones Avanzadas")
        # Activando clasificación múltiple (aún por implementar la lógica completa)
        show_batch = st.checkbox("Clasificación de múltiples imágenes (Beta)")

        st.markdown("---")
        st.markdown("Desarrollado con **keras 🧠**")

    # --- Clasificación individual ---
    st.subheader("📷 Sube una imagen del residuo")

    # Eliminamos las columnas y el selector de ejemplos. El uploader ocupa todo el ancho.
    archivo_subido = st.file_uploader("Arrastra y suelta tu imagen aquí o haz clic para subir", type=["jpg", "jpeg", "png"])
    
    imagen_a_procesar = None
    imagen_info_display = None # Para mostrar info sobre la imagen (subida)

    if archivo_subido:
        imagen_a_procesar = Image.open(archivo_subido).convert("RGB")
        imagen_info_display = "🖼️ Imagen cargada"

    if imagen_a_procesar:
        st.image(imagen_a_procesar, caption=imagen_info_display, use_column_width=True)

        st.markdown("---")
        if st.button("✨ ¡Clasificar Ahora! ✨"):
            # Preprocesamiento
            imagen_redim = imagen_a_procesar.resize((224, 224))
            img_array = np.expand_dims(np.array(imagen_redim) / 255.0, axis=0)

            with st.spinner("Analizando la imagen..."):
                time.sleep(1) # Simula un pequeño retraso para ver el spinner
                pred = modelo.predict(img_array)
                clase_predicha = clases_residuos[np.argmax(pred)]
                confianza = np.max(pred) * 100
                tipo = tipo_residuo.get(clase_predicha, "Desconocido")

            # Guardar en historial
            st.session_state["historial"].append(f"{clase_predicha} ({confianza:.2f}%) - {tipo}")

            # Resultado
            st.markdown(
                f"""
                <div class='result-box'>
                    <h3>🔍 Resultado de la clasificación:</h3>
                    <h2 style='color:#0d47a1;'>Clase: <span style='font-size:1.5em;'>{clase_predicha.upper()}</span></h2>
                    <h4>Probabilidad: <strong style='color:#388e3c;'>{confianza:.2f}%</strong></h4>
                    <h4>🧩 Tipo de residuo: <strong>{tipo}</strong> {'♻️' if tipo == 'Reciclable' else '🗑️'}</h4>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Información detallada en un expander
            if clase_predicha in info_detalle_clase:
                st.markdown(f"<div class='info-card'>{info_detalle_clase[clase_predicha]}</div>", unsafe_allow_html=True)

            # Gráfico de barras
            st.subheader("📊 Distribución de probabilidades")
            fig = px.bar(x=clases_residuos, y=pred[0], color=clases_residuos, title="Confianza por clase") # Usando Plotly Express
            fig.update_layout(yaxis_range=[0, 1]) # Establecer el rango del eje Y de 0 a 1
            st.plotly_chart(fig)

            # Botón de descarga
            st.download_button(
                label="📥 Descargar resultado",
                data=f"Clase: {clase_predicha}\nConfianza: {confianza:.2f}%\nTipo de residuo: {tipo}\nConsejo: {info_detalle_clase.get(clase_predicha, 'No hay consejos específicos.')}",
                file_name="prediccion_residuo.txt",
                mime="text/plain",
            )
            
            # --- Sección de Feedback ---
            st.markdown("---")
            st.subheader("👍 ¿Fue útil esta clasificación?")
            col_feedback_yes, col_feedback_no = st.columns(2)
            with col_feedback_yes:
                if st.button("Sí, fue correcto ✅", key="feedback_yes"):
                    st.success("¡Gracias por tu feedback! Nos ayuda a mejorar.")
            with col_feedback_no:
                if st.button("No, hubo un error ❌", key="feedback_no"):
                    st.warning("Lo sentimos. Estamos trabajando para mejorar la precisión del modelo.")
                    st.text_area("Opcional: ¿Qué clase esperabas?", key="feedback_text_area")
                    st.button("Enviar feedback detallado", key="send_detailed_feedback") # Podrías integrar esto con un sistema de logging


# # --- Información Educativa ---
with pestana_info:
    st.title("📘 Información para una correcta separación de residuos")
    st.markdown(
        """
        La correcta separación de residuos es fundamental para proteger nuestro planeta.
        Ayuda a **reducir la contaminación**, **ahorrar recursos** y **disminuir la cantidad de desechos** que van a los vertederos.
        """
    )

    st.subheader("♻️ Residuos Reciclables (Ejemplos comunes)")
    st.markdown(
        """
        <div class='info-card'>
            <p><strong>✅ Cartón:</strong> Cajas de cereales, cajas de zapatos, tubos de papel higiénico (limpios y secos).</p>
            <p><strong>✅ Vidrio:</strong> Botellas de bebidas, frascos de conservas (limpios, sin tapas de metal o plástico).</p>
            <p><strong>✅ Metal:</strong> Latas de refresco, latas de conserva, papel de aluminio (limpios y sin restos de comida).</p>
            <p><strong>✅ Papel:</strong> Periódicos, revistas, papel de oficina, sobres (sin ventanas de plástico, sin adhesivos excesivos).</p>
            <p><strong>✅ Plástico:</strong> Botellas de agua, envases de leche, envases de productos de limpieza (busca los símbolos de reciclaje ♻️ con los números 1 y 2).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("🗑️ Residuos Inorgánicos No Reciclables")
    st.markdown(
        """
        <div class='info-card red'>
            <p><strong>❌ Papel sucio/grasoso:</strong> Servilletas usadas, papel de cocina con aceite, cajas de pizza con grasa.</p>
            <p><strong>❌ Vidrios rotos/especiales:</strong> Cerámica, espejos, bombillas, vasos de cristal (riesgo de seguridad y composición diferente).</p>
            <p><strong>❌ Plásticos laminados/sucios:</strong> Envases de yogurt, envoltorios de golosinas, bolsas de snacks (generalmente no se reciclan).</p>
            <p><strong>❌ Otros:</strong> Juguetes rotos, pañales, toallas sanitarias, pilas (requieren desecho especial), cables, telgopor (EPS).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("🌱 ¿Y los residuos orgánicos?")
    st.markdown(
        """
        <div class='info-card yellow'>
            <p><strong>Compostables y biodegradables.</strong> Nuestro modelo no los clasifica actualmente, pero son clave para el compostaje:</p>
            <p><strong>🥕</strong> Cáscaras de frutas y verduras, restos de comida, posos de café, bolsitas de té, restos de poda ligera.</p>
            <p><strong>💡 ¡Importante!</strong> Evita mezclar residuos orgánicos con reciclables para no contaminar los materiales útiles. Si puedes hacer compost, ¡es una excelente forma de cerrar el ciclo!</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.info(
        """
        🌍 **Pequeñas acciones, grandes impactos.** Cada vez que separas correctamente un residuo, contribuyes a un futuro más verde y sostenible.
        """
    )

# --- Historial de Clasificaciones ---
with pestana_historial:
    st.title("📜 Historial de Clasificaciones")
    if st.session_state["historial"]:
        # Botón para limpiar el historial
        if st.button("🗑️ Borrar Historial", key="clear_history_button"):
            st.session_state["historial"] = []
            st.success("Historial borrado.")
            st.experimental_rerun() # Recargar la página para mostrar el historial vacío
        
        st.markdown("---")
        for item in reversed(st.session_state["historial"]): # Mostrar los más recientes primero
            st.write(f"✅ {item}")
    else:
        st.info("Aún no has realizado ninguna clasificación. ¡Sube una imagen para empezar!")

# --- Preguntas Frecuentes (Nueva Pestaña) ---
with pestana_faq:
    st.title("❓ Preguntas Frecuentes sobre el Clasificador y Reciclaje")

    st.markdown("---")

    with st.expander("¿Qué tipo de residuos puede clasificar este modelo?"):
        st.write("Actualmente, el modelo está entrenado para clasificar residuos en 6 categorías principales: **cartón, vidrio, metal, papel, plástico** y **basura** (para inorgánicos no reciclables). También indica si el residuo es general **reciclable** o **inorgánico**.")

    with st.expander("¿Qué tan preciso es el modelo?"):
        st.write("El modelo tiene una **precisión general (accuracy) del 73%**. Esto significa que, en promedio, acierta en la clasificación de 73 de cada 100 imágenes. Las métricas de Macro Promedio y Promedio Ponderado te dan una idea más detallada de su rendimiento en cada clase.")

    with st.expander("¿Puedo subir varias imágenes a la vez?"):
        st.write("Hemos incluido una opción 'Clasificación de múltiples imágenes (Beta)' en la barra lateral. Esta funcionalidad está en desarrollo y se implementará completamente en futuras actualizaciones para permitir el procesamiento por lotes.")

    with st.expander("¿Qué hago si el modelo clasifica mi residuo incorrectamente?"):
        st.write("¡Agradecemos tu feedback! Al final de cada clasificación, encontrarás una sección para indicar si la predicción fue correcta o no. Tu información nos ayuda a identificar áreas de mejora y, potencialmente, a entrenar el modelo con más datos.")

    with st.expander("¿Qué pasa con los residuos orgánicos?"):
        st.write("Este modelo se enfoca principalmente en residuos inorgánicos. Los residuos orgánicos (restos de comida, cáscaras, etc.) son ideales para el **compostaje**. Recomendamos tener un contenedor separado para ellos si es posible.")

    with st.expander("¿Por qué es importante lavar y secar los materiales reciclables?"):
        st.write("Los restos de comida y líquidos pueden contaminar otros materiales reciclables, haciendo que no puedan ser procesados. Lavar y secar ayuda a asegurar que los materiales mantengan su valor y puedan ser reciclados de manera efectiva.")

    st.markdown("---")
    st.info("¿No encontraste lo que buscabas? ¡Envíanos tus preguntas o sugerencias!")

# --- Nueva Pestaña: Acerca de ---
with pestana_about:
    st.title("ℹ️ Acerca del Proyecto 'Clasificador de Residuos Inteligente'")
    st.markdown(
        """
        Este proyecto es una aplicación web interactiva desarrollada con **Streamlit** que utiliza un modelo de **machine learning, entrenado con Keras**
        para clasificar imágenes de residuos. Su objetivo principal es educar y asistir a los usuarios en la correcta separación de desechos,
        promoviendo así prácticas de reciclaje más eficientes y sostenibles.
        """
    )

    st.subheader("🚀 Objetivos del Proyecto")
    st.markdown(
        """
        * **Educación Ambiental:** Proveer información clara y accesible sobre cómo clasificar diferentes tipos de residuos.
        * **Asistencia en el Reciclaje:** Ofrecer una herramienta práctica para identificar rápidamente la categoría de un residuo.
        * **Concientización:** Fomentar una cultura de reciclaje y responsabilidad ambiental.
        """
    )

    st.subheader("🧠 El Modelo de IA")
    st.markdown(
        """
        El corazón de esta aplicación es un modelo de **aprendizaje profundo** entrenado para reconocer patrones en imágenes de residuos.
        Fue entrenado con un dataset diverso para identificar las categorías de cartón, vidrio, metal, papel, plástico y basura.
        Las métricas de rendimiento del modelo (precisión, macro promedio, promedio ponderado) se muestran en la barra lateral de la pestaña 'Clasificador'.
        """
    )

    st.subheader("🛠️ Tecnologías Utilizadas")
    st.markdown(
        """
        * **Streamlit:** Para el desarrollo rápido de la interfaz de usuario web.
        * **Keras:** Para la construcción, entrenamiento y despliegue del modelo de clasificación de imágenes.
        * **NumPy:** Para el procesamiento numérico de imágenes.
        * **Pillow (PIL):** Para la manipulación de imágenes.
        * **Plotly Express:** Para la visualización interactiva de los resultados de probabilidad.
        * **Requests:** Para la descarga de imágenes de ejemplo (cuando estaban presentes) y recursos externos.
        """
    )

    st.subheader("👥 Equipo y Colaboración")
    st.markdown(
        """
        Este proyecto fue desarrollado como parte de un esfuerzo por aplicar la ciencia de datos a problemas del mundo real.
        Estamos abiertos a sugerencias y colaboraciones para mejorar la precisión del modelo y añadir nuevas funcionalidades.
        """
    )
    st.markdown("---")
    st.info("¡Trabajando juntos por un planeta más limpio!")

# --- Créditos ---
st.sidebar.markdown("---")
st.sidebar.text("⚡ Creado con Streamlit, keras y Plotly")