import streamlit as st
import numpy as np
import plotly.express as px
from PIL import Image
import requests
from io import BytesIO
import time
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import tempfile
import os

# --- Configuración de página ---
st.set_page_config(
    page_title="Clasificador de Residuos Inteligente",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/streamlit',
        'Report a bug': "https://github.com/streamlit",
        'About': "### ♻️ Clasificador de Residuos con IA\nAplicación desarrollada con Keras,TensorFlow y Streamlit"
    }
)

# --- Cargar modelo ---
@st.cache_resource
def cargar_modelo():
    with st.spinner("Cargando modelo de IA... Esto puede tardar un momento."):
        try:
            model = tf.keras.models.load_model("modelo_residuos.keras")
            st.success("¡Modelo cargado con éxito!")
            return model
        except Exception as e:
            st.error(f"Error al cargar el modelo: {e}. Asegúrate de que 'modelo_residuos.keras' esté en la misma carpeta.")
            st.stop()

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
    'cartón': {
        "descripcion": "El cartón debe estar limpio y seco. Dóblalo para ahorrar espacio y quita cualquier residuo de comida o cinta adhesiva excesiva.",
        "consejos": [
            "Las cajas de pizza grasosas no van al reciclaje",
            "Los vasos de café y cartones de leche/jugo plastificados no son reciclables"
        ],
        "icono": "📦"
    },
    'vidrio': {
        "descripcion": "Las botellas y frascos de vidrio (transparente, verde, ámbar) son reciclables. Lávalos bien y quita tapas y etiquetas.",
        "consejos": [
            "El vidrio roto, espejos, bombillas y cerámica no se reciclan aquí",
            "Los vasos de cristal tienen diferente composición"
        ],
        "icono": "🍾"
    },
    'metal': {
        "descripcion": "Las latas de aluminio y acero, así como envases de alimentos (limpios) y aerosoles vacíos son reciclables.",
        "consejos": [
            "Asegúrate de que no contengan líquidos",
            "Metales grandes requieren centros de acopio específicos"
        ],
        "icono": "🥫"
    },
    'papel': {
        "descripcion": "El papel blanco o de oficina, periódicos, revistas, folletos son reciclables. Debe estar limpio y seco.",
        "consejos": [
            "Evita papel encerado, de fotos o con adhesivos",
            "Servilletas y toallas de papel usadas no son reciclables"
        ],
        "icono": "📄"
    },
    'plástico': {
        "descripcion": "Revisa el símbolo de reciclaje (triángulo con un número). Los plásticos PET (1) y HDPE (2) son los más aceptados.",
        "consejos": [
            "Límpialos y aplástalos para ahorrar espacio",
            "Otros plásticos (3-7) a menudo no son reciclables"
        ],
        "icono": "🧴"
    },
    'basura': {
        "descripcion": "Residuos inorgánicos no reciclables como papel sucio, toallas sanitarias, pañales, cerámica rota, etc.",
        "consejos": [
            "Deben ir al vertedero o incineración",
            "Considera reducir el consumo de estos productos"
        ],
        "icono": "🗑️"
    }
}

# --- Estilos personalizados ---
st.markdown(
    """
    <style>
    .main {
        background-color: #f4f6f8;
    }
    .stButton > button {
        color: white;
        background-color: #28a745;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #218838;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .result-box {
        padding: 25px;
        background: linear-gradient(135deg, #e0f2f7, #ffffff);
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
        animation: fadeIn 1s ease-out;
        border: 1px solid #e0e0e0;
    }
    .result-box h2 {
        color: #0d47a1;
        font-size: 2.2em;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fff8, #e8f5e9);
        padding-top: 20px;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .info-card {
        background: white;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .info-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .info-card.red {
        border-left: 5px solid #dc3545;
    }
    .info-card.yellow {
        border-left: 5px solid #ffc107;
    }
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
    .stFileUploader {
        border: 2px dashed #28a745;
        border-radius: 12px;
        padding: 20px;
        background-color: rgba(40, 167, 69, 0.05);
    }
    .stFileUploader:hover {
        background-color: rgba(40, 167, 69, 0.1);
    }
    .progress-bar {
        height: 10px;
        background-color: #e0e0e0;
        border-radius: 5px;
        margin: 10px 0;
        overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        width: 0%;
        transition: width 0.5s ease;
    }
    .confidence-meter {
        display: flex;
        align-items: center;
        margin: 10px 0;
    }
    .confidence-label {
        width: 100px;
        font-weight: bold;
    }
    .confidence-value {
        margin-left: 10px;
        font-weight: bold;
        color: #388e3c;
    }
    .material-icon {
        font-size: 24px;
        margin-right: 10px;
        vertical-align: middle;
    }
    .stMarkdown h1 {
        background: linear-gradient(45deg, #28a745, #0d47a1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
        padding-bottom: 10px;
    }
    .stImage img {
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        max-height: 400px;
        object-fit: contain;
    }
    .stImage img:hover {
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
        margin-left: 8px;
    }
    .badge-recyclable {
        background-color: #4CAF50;
        color: white;
    }
    .badge-nonrecyclable {
        background-color: #F44336;
        color: white;
    }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
        color: #28a745;
    }
    .feature-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        height: 100%;
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    .carbon-calculator {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .carbon-result {
        font-size: 1.5rem;
        font-weight: bold;
        color: #388e3c;
        margin-top: 10px;
    }
    .confetti {
        position: fixed;
        width: 10px;
        height: 10px;
        background-color: #f00;
        border-radius: 50%;
        animation: fall 5s linear infinite;
    }
    @keyframes fall {
        0% {
            transform: translateY(-100vh) rotate(0deg);
            opacity: 1;
        }
        100% {
            transform: translateY(100vh) rotate(360deg);
            opacity: 0;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Historial de clasificaciones ---
if "historial" not in st.session_state:
    st.session_state["historial"] = []

# --- Contador de clasificaciones ---
if "contador_clasificaciones" not in st.session_state:
    st.session_state["contador_clasificaciones"] = 0

# --- Pestañas ---
pestana_clasificador, pestana_info, pestana_historial, pestana_faq, pestana_about = st.tabs(
    ["🧠 Clasificador", "📘 Información Educativa", "📜 Historial", "❓ Preguntas Frecuentes", "ℹ️ Acerca de"]
)

# --- Función para mostrar confeti ---
def show_confetti():
    st.markdown("""
    <script>
    function createConfetti() {
        const colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff'];
        for (let i = 0; i < 50; i++) {
            const confetti = document.createElement('div');
            confetti.className = 'confetti';
            confetti.style.left = Math.random() * 100 + 'vw';
            confetti.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
            confetti.style.animationDuration = (Math.random() * 3 + 2) + 's';
            confetti.style.width = (Math.random() * 10 + 5) + 'px';
            confetti.style.height = confetti.style.width;
            document.body.appendChild(confetti);
            
            setTimeout(() => {
                confetti.remove();
            }, 5000);
        }
    }
    createConfetti();
    </script>
    """, unsafe_allow_html=True)

# --- Función para preprocesar imagen ---
def preprocess_image(img, target_size=(224, 224)):
    # Convertir a RGB si es necesario
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Redimensionar
    img = img.resize(target_size)
    
    # Convertir a array y normalizar
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array

# --- Función para clasificar imagen ---
def classify_image(img, model):
    # Preprocesamiento
    img_array = preprocess_image(img)
    
    # Predicción
    pred = model.predict(img_array)
    clase_predicha = clases_residuos[np.argmax(pred)]
    confianza = np.max(pred) * 100
    tipo = tipo_residuo.get(clase_predicha, "Desconocido")
    
    return clase_predicha, confianza, tipo, pred[0]

# --- Pestaña Clasificador ---
with pestana_clasificador:
    st.title("🌍 Clasificador Inteligente de Residuos")
    st.markdown(
        """
        Este sistema utiliza **Inteligencia Artificial con Keras, TensorFlow** para identificar residuos y clasificarlos como **reciclables** o **inorgánicos**.
        Sube una imagen y descubre a qué categoría pertenece tu residuo: **cartón, vidrio, metal, papel, plástico o basura**.
        """
    )
    
    # --- Sidebar ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/927/927567.png", width=120)
        st.header("📈 Métricas del Modelo")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Precisión", "73%", "3% desde la última versión")
        with col2:
            st.metric("Recall", "70%", "2% desde la última versión")
        
        st.progress(0.73, text="Rendimiento del modelo")
        
        # Selector de umbral de confianza
        umbral_confianza = 73
        
        st.markdown("---")
        st.markdown("Desarrollado con **Keras, TensorFlow 🧠 y Streamlit**")

    # --- Sección de carga de imágenes ---
    st.subheader("📷 Sube una imagen del residuo")
    
    # Opciones de entrada
    input_method = st.radio("Selecciona método de entrada:", 
                          ["Subir imagen", "Tomar foto con cámara"],
                          horizontal=True)
    
    imagen_a_procesar = None
    imagen_info_display = None
    
    if input_method == "Subir imagen":
        archivo_subido = st.file_uploader("Arrastra y suelta tu imagen aquí o haz clic para subir", 
                                        type=["jpg", "jpeg", "png", "webp"], 
                                        key="file_uploader")
        if archivo_subido:
            imagen_a_procesar = Image.open(archivo_subido)
            imagen_info_display = "🖼️ Imagen cargada desde archivo"
    
    elif input_method == "Tomar foto con cámara":
        foto_camara = st.camera_input("Toma una foto del residuo")
        if foto_camara:
            imagen_a_procesar = Image.open(foto_camara)
            imagen_info_display = "📸 Foto tomada con cámara"
    
    # Mostrar imagen si está cargada
    if imagen_a_procesar:
        st.image(imagen_a_procesar, caption=imagen_info_display, use_column_width=True)

        # Botón de clasificación
        st.markdown("---")
        if st.button("✨ ¡Clasificar Ahora! ✨", use_container_width=True):
            with st.spinner("Analizando la imagen..."):
                # Clasificar la imagen
                clase_predicha, confianza, tipo, pred = classify_image(imagen_a_procesar, modelo)
                
                # Actualizar contador
                st.session_state["contador_clasificaciones"] += 1
                
                # Guardar en historial
                registro = {
                    "clase": clase_predicha,
                    "confianza": confianza,
                    "tipo": tipo,
                    "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "imagen": imagen_a_procesar
                }
                st.session_state["historial"].append(registro)
                
                # Mostrar resultados
                st.markdown(
                    f"""
                    <div class='result-box'>
                        <h3>🔍 Resultado de la clasificación:</h3>
                        <h2>Clase: {clase_predicha.upper()} {info_detalle_clase[clase_predicha]['icono']}</h2>
                        <h4>Probabilidad: <strong style='color:#388e3c;'>{confianza:.2f}%</strong></h4>
                        <h4>🧩 Tipo de residuo: <strong>{tipo}</strong> 
                            <span class='badge {'badge-recyclable' if tipo == 'Reciclable' else 'badge-nonrecyclable'}'>
                                {'♻️ Reciclable' if tipo == 'Reciclable' else '🗑️ No reciclable'}
                            </span>
                        </h4>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                # Mostrar confeti si la confianza es alta
                if confianza > 90:
                    show_confetti()
                    st.balloons()
                
                # Información detallada en tarjeta
                with st.expander(f"📌 Información detallada sobre {clase_predicha}", expanded=True):
                    st.markdown(f"### {info_detalle_clase[clase_predicha]['icono']} {clase_predicha.capitalize()}")
                    st.markdown(info_detalle_clase[clase_predicha]["descripcion"])
                    
                    st.markdown("**💡 Consejos importantes:**")
                    for consejo in info_detalle_clase[clase_predicha]["consejos"]:
                        st.markdown(f"- {consejo}")
                    
                    if tipo == "Reciclable":
                        st.success("✅ Este material puede ser reciclado. Asegúrate de limpiarlo y depositarlo en el contenedor adecuado.")
                    else:
                        st.warning("⚠️ Este material no es reciclable. Deposítalo en el contenedor de basura general.")
                
                # Gráfico de barras interactivo
                st.subheader("📊 Distribución de probabilidades")
                
                # Crear dataframe para Plotly
                import pandas as pd
                df_pred = pd.DataFrame({
                    "Clase": clases_residuos,
                    "Probabilidad": pred,
                    "Tipo": [tipo_residuo[clase] for clase in clases_residuos]
                })
                
                # Gráfico interactivo
                fig = px.bar(df_pred, x="Clase", y="Probabilidad", color="Tipo",
                            color_discrete_map={"Reciclable": "#4CAF50", "Inorgánico": "#F44336"},
                            hover_data=["Probabilidad"],
                            labels={"Probabilidad": "Probabilidad (%)", "Clase": "Categoría de residuo"},
                            title="Confianza de predicción por categoría")
                
                fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar métricas de confianza
                st.subheader("📈 Nivel de confianza")
                st.markdown(f"""
                    <div class="confidence-meter">
                        <div class="confidence-label">Confianza:</div>
                        <div class="progress-bar">
                            <div class="progress-bar-fill" style="width: {confianza}%"></div>
                        </div>
                        <div class="confidence-value">{confianza:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
                
                if confianza < umbral_confianza:
                    st.warning("⚠️ La confianza en esta predicción es baja. Considera verificar manualmente la clasificación.")
                
                # Calculadora de impacto ambiental
                with st.expander("🌍 Calculadora de impacto ambiental"):
                    st.markdown("""
                        <div class="carbon-calculator">
                            <h4>♻️ Impacto positivo potencial</h4>
                            <p>Al reciclar correctamente este material, podrías estar contribuyendo a:</p>
                    """, unsafe_allow_html=True)
                    
                    if tipo == "Reciclable":
                        beneficios = {
                            "cartón": {"ahorro": "17 árboles", "energia": "4,000 kWh", "agua": "7,000 galones"},
                            "vidrio": {"ahorro": "30% de energía", "emisiones": "20% menos CO2"},
                            "metal": {"ahorro": "74% de energía", "recursos": "1.5 toneladas de mineral"},
                            "papel": {"ahorro": "4,000 kWh", "agua": "7,000 galones", "arboles": "17 árboles"},
                            "plástico": {"ahorro": "5.774 kWh", "petroleo": "16.3 barriles"}
                        }
                        
                        if clase_predicha in beneficios:
                            st.markdown("<ul>", unsafe_allow_html=True)
                            for key, value in beneficios[clase_predicha].items():
                                st.markdown(f"<li>Ahorrar <strong>{value}</strong> por tonelada reciclada</li>", unsafe_allow_html=True)
                            st.markdown("</ul>", unsafe_allow_html=True)
                        
                        st.markdown("""
                            <div class="carbon-result">
                                ¡Buen trabajo! Estás ayudando a reducir la huella de carbono.
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <p>Este material no es reciclable, pero al clasificarlo correctamente evitas que contamine otros materiales reciclables.</p>
                            <div class="carbon-result">
                                Considera reducir el consumo de este tipo de productos.
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Opciones de descarga
                st.markdown("---")
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    # Descargar resultado como texto
                    st.download_button(
                        label="📥 Descargar resultado (TXT)",
                        data=f"""Clasificación de residuo:
- Clase: {clase_predicha}
- Probabilidad: {confianza:.2f}%
- Tipo: {tipo}
- Fecha: {time.strftime("%Y-%m-%d %H:%M:%S")}

Consejos:
{info_detalle_clase[clase_predicha]["descripcion"]}

""" + "\n".join(f"- {c}" for c in info_detalle_clase[clase_predicha]["consejos"]),
        file_name=f"clasificacion_{clase_predicha}.txt",
        mime="text/plain",
    )
                
                with col_dl2:
                    # Descargar imagen con anotación
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                        # Crear imagen con anotación
                        img_annotated = np.array(imagen_a_procesar.copy())
                        img_annotated = cv2.cvtColor(img_annotated, cv2.COLOR_RGB2BGR)
                        
                        # Añadir texto
                        text = f"{clase_predicha} ({confianza:.1f}%)"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(img_annotated, text, (20, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        
                        # Guardar temporalmente
                        cv2.imwrite(tmpfile.name, img_annotated)
                        
                        # Botón de descarga
                        with open(tmpfile.name, "rb") as file:
                            st.download_button(
                                label="📷 Descargar imagen anotada",
                                data=file,
                                file_name=f"clasificado_{clase_predicha}.png",
                                mime="image/png",
                            )
                    
                    # Eliminar archivo temporal
                    os.unlink(tmpfile.name)

# --- Pestaña Información Educativa ---
with pestana_info:
    st.title("📘 Información para una correcta separación de residuos")
    st.markdown(
        """
        La correcta separación de residuos es fundamental para proteger nuestro planeta.
        Ayuda a **reducir la contaminación**, **ahorrar recursos** y **disminuir la cantidad de desechos** que van a los vertederos.
        """
    )
    
    # Tarjetas de información
    st.subheader("♻️ Guía Rápida de Reciclaje")
    
    cols = st.columns(3)
    for i, (clase, info) in enumerate(info_detalle_clase.items()):
        with cols[i % 3]:
            tipo = tipo_residuo[clase]
            st.markdown(f"""
                <div class="feature-card">
                    <div style="font-size: 2rem; text-align: center;">{info['icono']}</div>
                    <h3 style="text-align: center;">{clase.capitalize()}</h3>
                    <p><strong>Tipo:</strong> <span class="badge {'badge-recyclable' if tipo == 'Reciclable' else 'badge-nonrecyclable'}">
                        {tipo}
                    </span></p>
                    <p>{info['descripcion'].split('.')[0]}.</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Sección de beneficios
    st.markdown("---")
    st.subheader("🌎 Beneficios Ambientales del Reciclaje")
    
    beneficios = [
        {"icon": "🌳", "title": "Ahorro de recursos", "desc": "Reciclar una tonelada de papel salva 17 árboles y ahorra 26,500 litros de agua."},
        {"icon": "⚡", "title": "Ahorro de energía", "desc": "Reciclar aluminio usa 95% menos energía que producirlo nuevo."},
        {"icon": "🏭", "title": "Reducción de emisiones", "desc": "El reciclaje reduce las emisiones de gases de efecto invernadero."},
        {"icon": "🗑️", "title": "Menos vertederos", "desc": "Cada material reciclado es menos basura en vertederos e incineradoras."},
        {"icon": "💼", "title": "Creación de empleos", "desc": "La industria del reciclaje genera 10 veces más empleos que los vertederos."},
        {"icon": "💰", "title": "Ahorro económico", "desc": "Reciclar es más barato que recolectar y disponer de basura tradicionalmente."}
    ]
    
    cols = st.columns(3)
    for i, beneficio in enumerate(beneficios):
        with cols[i % 3]:
            st.markdown(f"""
                <div class="feature-card">
                    <div style="font-size: 2rem;">{beneficio['icon']}</div>
                    <h4>{beneficio['title']}</h4>
                    <p>{beneficio['desc']}</p>
                </div>
            """, unsafe_allow_html=True)
    
# --- Pestaña Historial ---
with pestana_historial:
    st.title("📜 Historial de Clasificaciones")
    
    if st.session_state["historial"]:
        # Estadísticas del historial
        st.subheader("📊 Estadísticas")
        
        # Calcular métricas
        total_clasificaciones = len(st.session_state["historial"])
        reciclables = sum(1 for item in st.session_state["historial"] if item["tipo"] == "Reciclable")
        no_reciclables = total_clasificaciones - reciclables
        avg_confianza = sum(item["confianza"] for item in st.session_state["historial"]) / total_clasificaciones
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de clasificaciones", total_clasificaciones)
        col2.metric("Materiales reciclables", f"{reciclables} ({reciclables/total_clasificaciones:.0%})")
        col3.metric("Confianza promedio", f"{avg_confianza:.1f}%")
        
        # Gráfico de distribución
        fig = px.pie(
            names=["Reciclables", "No reciclables"],
            values=[reciclables, no_reciclables],
            color=["Reciclables", "No reciclables"],
            color_discrete_map={"Reciclables": "#4CAF50", "No reciclables": "#F44336"},
            title="Distribución de tus clasificaciones"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📝 Registro detallado")
        
        # Mostrar historial en orden inverso (más reciente primero)
        for idx, item in enumerate(reversed(st.session_state["historial"])):
            with st.expander(f"{idx+1}. {item['clase']} - {item['fecha']}", expanded=idx==0):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(item["imagen"], caption=f"Imagen clasificada", width=200)
                
                with col2:
                    st.markdown(f"""
                        **Clase:** {item['clase'].capitalize()}  
                        **Confianza:** {item['confianza']:.1f}%  
                        **Tipo:** <span class="badge {'badge-recyclable' if item['tipo'] == 'Reciclable' else 'badge-nonrecyclable'}">
                            {item['tipo']}
                        </span>  
                        **Fecha:** {item['fecha']}
                    """, unsafe_allow_html=True)
                    
                    if item['clase'] in info_detalle_clase:
                        st.markdown("**Consejos:**")
                        for consejo in info_detalle_clase[item['clase']]["consejos"]:
                            st.markdown(f"- {consejo}")
    else:
        st.info("Aún no has realizado ninguna clasificación. ¡Sube una imagen para empezar!")
        st.image("https://cdn-icons-png.flaticon.com/512/4076/4076478.png", width=300)

# --- Pestaña Preguntas Frecuentes ---
with pestana_faq:
    st.title("❓ Preguntas Frecuentes sobre el Clasificador y Reciclaje")
    
    # Organizado por categorías
    tab_uso, tab_tecnico, tab_reciclaje = st.tabs(["Uso de la App", "Aspectos Técnicos", "Preguntas sobre Reciclaje"])
    
    with tab_uso:
        st.markdown("""
        ### 🧩 ¿Cómo uso el clasificador?
        1. Sube una foto clara del residuo que quieres clasificar
        2. Ajusta la imagen si es necesario (rotación, brillo, contraste)
        3. Haz clic en "Clasificar Ahora"
        4. Revisa los resultados y consejos de reciclaje
        
        ### 📷 ¿Qué tipo de imágenes funcionan mejor?
        - Fotos claras y bien iluminadas
        - Enfoca directamente el objeto
        - Evita fondos muy ocupados
        - Muestra el objeto desde varios ángulos si no estás seguro
        
        ### 🔍 ¿Qué hago si la clasificación es incorrecta?
        Usa la sección de feedback al final de los resultados para informarnos. Esto nos ayuda a mejorar el modelo.
        """)
    
    with tab_tecnico:
        st.markdown("""
        ### 🤖 ¿Cómo funciona el modelo de Clasificación?
        Usamos una red neuronal convolucional (CNN) entrenada con miles de imágenes de residuos. 
        El modelo analiza patrones visuales para predecir la categoría más probable.
        
        ### 📊 ¿Qué significan las métricas del modelo?
        - **Precisión (73%):** Porcentaje de clasificaciones correctas
        - **Recall (70%):** Capacidad para identificar todos los casos positivos
        - **Confianza:** Certeza del modelo en cada predicción
        
        ### 🚀 ¿Cómo puedo contribuir al proyecto?
        - Reportando errores de clasificación
        - Compartiendo la app con más personas
        - Proporcionando imágenes etiquetadas para entrenamiento
        """)
    
    with tab_reciclaje:
        st.markdown("""
        ### ♻️ ¿Por qué es importante separar los residuos?
        - Reduce la contaminación
        - Ahorra recursos naturales
        - Disminuye la cantidad de basura en vertederos
        - Genera empleos en la industria del reciclaje
        
        ### 🏡 ¿Cómo empezar a reciclar en casa?
        1. Consigue contenedores separados
        2. Educa a todos en el hogar
        3. Limpia los materiales reciclables
        4. Consulta las normas locales
        
        ### 🌎 ¿Qué impacto tiene el reciclaje?
        - Reciclar una lata de aluminio ahorra energía para 3 horas de TV
        - Cada tonelada de papel reciclado salva 17 árboles
        - El vidrio se recicla infinitamente sin perder calidad
        """)
    
# --- Pestaña Acerca de ---
with pestana_about:
    st.title("ℹ️ Acerca del Proyecto")
    
    st.markdown("""
    ## ♻️ Clasificador de Residuos Inteligente
    
    Una aplicación web interactiva desarrollada con **Keras, TensorFlow y Streamlit** que utiliza 
    **aprendizaje profundo** para clasificar imágenes de residuos y promover el reciclaje correcto.
    """)
    
    # Equipo y colaboradores
    st.markdown("---")
    st.subheader("👥 Equipo")
    
    cols = st.columns(5)
    miembros = [
        {"nombre": "Vasquez Gonzales Bruno Joel"},
        {"nombre": "Almendro Torrico Fabian Israel"},
        {"nombre": "Choquerive Ramos Richard"},
        {"nombre": "Yupanqui Villarroel Simon Tito"},
        {"nombre": "Solano Nieto Benjamin"}
    ]
    
    for i, miembro in enumerate(miembros):
        with cols[i]:
            st.markdown(f"""
                <div class="feature-card">
                    <h3>{miembro['nombre']}</h3>
                </div>
            """, unsafe_allow_html=True)
    
    # Tecnologías utilizadas
    st.markdown("---")
    st.subheader("🛠️ Tecnologías Utilizadas")
    
    tech_cols = st.columns(4)
    tecnologias = [
        {"nombre": "TensorFlow", "icono": "https://cdn-icons-png.flaticon.com/512/5961/5961202.png", "uso": "Modelo de clasificación de imágenes"},
        {"nombre": "Streamlit", "icono": "https://streamlit.io/images/brand/streamlit-mark-color.png", "uso": "Interfaz web interactiva"},
        {"nombre": "Plotly", "icono": "https://images.plot.ly/logo/new-branding/plotly-logomark.png", "uso": "Visualización de datos"},
        {"nombre": "OpenCV", "icono": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/OpenCV_Logo_with_text_svg_version.svg/1200px-OpenCV_Logo_with_text_svg_version.svg.png", "uso": "Procesamiento de imágenes"}
    ]
    
    for i, tech in enumerate(tecnologias):
        with tech_cols[i % 4]:
            st.image(tech["icono"], width=50)
            st.markdown(f"**{tech['nombre']}**")
            st.caption(tech["uso"])
    
    # Hoja de ruta y contribuciones
    st.markdown("---")
    st.subheader("🚀 Hoja de Ruta Futura")
    
    roadmap = [
        {"version": "v2.0", "features": "Clasificación de múltiples imágenes simultáneas", "estado": "En desarrollo"},
        {"version": "v2.1", "features": "Detección de objetos en tiempo real con cámara", "estado": "Planificado"},
        {"version": "v3.0", "features": "Integración con sistemas municipales de reciclaje", "estado": "En investigación"},
        {"version": "v3.5", "features": "Reconocimiento de residuos orgánicos", "estado": "Planificado"}
    ]
    
    for item in roadmap:
        with st.expander(f"{item['version']}: {item['features']} ({item['estado']})"):
            st.markdown(f"**Estado actual:** {item['estado']}")
            if item['estado'] == "En desarrollo":
                st.progress(20, text="Progreso del desarrollo")
            elif item['estado'] == "Planificado":
                st.progress(10, text="En planificación")
            else:
                st.progress(15, text="Investigación en curso")
    
    # Cómo contribuir
    st.markdown("---")
    st.subheader("🤝 ¿Quieres contribuir?")
    
    st.markdown("""
    Este es un proyecto de código abierto. Puedes ayudar de varias formas:
    
    - 🌟 **Dándonos una estrella** en GitHub
    - 🐛 **Reportando errores** o problemas
    - 📸 **Proporcionando imágenes** etiquetadas para entrenamiento
    - 📢 **Compartiendo** la aplicación con tu comunidad
    
    Visita nuestro [repositorio en GitHub](https://github.com/bruno-vasquez/ReciclajeBasura.git) para más información.
    """)
    
    # Créditos finales
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; background-color: #e8f5e9; border-radius: 10px;">
        <h3>🌍 Trabajando juntos por un planeta más limpio</h3>
        <p>Cada pequeña acción cuenta. ¡Gracias por ser parte del cambio!</p>
    </div>
    """, unsafe_allow_html=True)
    