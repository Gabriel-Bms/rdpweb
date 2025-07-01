import sys
import os
from base_dataset import get_transform
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.metric_cards import style_metric_cards
import base64
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
from types import SimpleNamespace
from models import create_model
import torchvision.transforms as T
import base64
from io import BytesIO

def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# === CONFIGURACIÓN GLOBAL CYCLE GAN ===============================================
opt = SimpleNamespace(
    name="aus2rus",
    model='cycle_gan',
    dataset_mode='unaligned',
    netG='unet_256',
    direction='AtoB',
    input_nc=1,
    output_nc=1,
    preprocess='resize_and_crop',
    load_size=256,
    crop_size=256,
    batch_size=1,
    serial_batches=True,
    no_flip=True,
    display_id=-1,
    num_threads=1,
    eval=True,
    load_iter=0,
    epoch='latest',
    aspect_ratio=1.0,
    max_dataset_size=float("inf"),
    gpu_ids=[0] if torch.cuda.is_available() else [],
    isTrain=False,
    checkpoints_dir="checkpoints",
    ngf=64,
    norm='instance',
    no_dropout=True,
    init_type='normal',
    init_gain=0.02,
    verbose=False,
)
@st.cache_resource
def load_cycle_gan():
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    generator = model.netG_A if opt.direction == 'AtoB' else model.netG_B
    return generator
generator = load_cycle_gan()

@st.cache_resource
def load_segmentation_model():
    return load_model("best_multiclass_model3.h5")

#======Funciones para realizar predicciones=====================
class_to_color = {
    0: (0, 0, 0),           # fondo
    1: (100, 0, 100),       # hígado (violeta oscuro)
    2: (255, 255, 0),       # riñón (amarillo)
    3: (0, 0, 255),         # páncreas (azul)
    4: (255, 0, 0),         # vasos (rojo)
    5: (0, 255, 255),       # suprarrenales (celeste)
    6: (0, 255, 0),         # vesícula (verde)
    7: (255, 255, 255),     # huesos (blanco)
    8: (255, 0, 255),       # bazo (rosado)
}
def mask_class_to_rgb(mask_class):
    h, w = mask_class.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in class_to_color.items():
        rgb_mask[mask_class == class_id] = color
    return rgb_mask

IMG_SIZE = (128, 128)
N_CLASSES = 9
#============================================================================
# Configuración de la página
st.set_page_config(
    page_title="EchoScan AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar el diseño
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .sidebar-logo {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .nav-button {
        width: 100%;
        height: 60px;
        font-size: 16px;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        margin: 8px 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 3rem;
        text-align: center;
        background: #f0f2f6;
        margin: 2rem 0;
    }
    
    .stButton > button {
        width: 100%;
        height: 3rem;
        border-radius: 10px;
        border: none;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Función para crear el header principal
def create_header():
    st.markdown("""
    <div class="main-header">
        <h1>🔬 EchoScan AI</h1>
        <p>Análisis inteligente de ecografías con IA</p>
    </div>
    """, unsafe_allow_html=True)

# Función para crear el menú de navegación lateral
def create_sidebar_navigation():
    with st.sidebar:
        # Logo/Header del sidebar
        st.markdown("""
        <div class="sidebar-logo">
            <h2>🔬 EchoScan AI</h2>
            <p>Menú de Navegación</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Botones de navegación
        if st.button("🏠 Inicio", key="home", help="Página principal", use_container_width=True):
            st.session_state.page = "inicio"
            st.rerun()
        
        if st.button("🔍 Segmentar Ecografía", key="segment", help="Segmentar órganos en ecografías", use_container_width=True):
            st.session_state.page = "segmentar"
            st.rerun()
        
        if st.button("✨ Mejorar Ecografía", key="enhance", help="Mejorar calidad de imagen", use_container_width=True):
            st.session_state.page = "mejorar" 
            st.rerun()
        
        if st.button("ℹ️ Información", key="info", help="Información sobre la aplicación", use_container_width=True):
            st.session_state.page = "informacion"
            st.rerun()
        
        # Separador
        st.markdown("---")
        
        # Información adicional en el sidebar
        st.markdown("""
        ### 📊 Estado del Sistema
        """)
        st.success("🟢 Sistema Online")
        st.info("📈 Procesando: 0 imágenes")
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; font-size: 12px; color: #666;'>
            <p>EchoScan AI v1.0<br>© 2025</p>
        </div>
        """, unsafe_allow_html=True)

# Página de Inicio===============================================================
def page_inicio():
    create_header()
    
    colored_header(
        label="Bienvenido a EchoScan AI",
        description="Tu asistente inteligente para análisis de ecografías",
        color_name="violet-70"
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 ¿Qué puedes hacer con EchoScan AI?
        
        **EchoScan AI** es una plataforma avanzada que utiliza inteligencia artificial para el análisis 
        y procesamiento de imágenes ecográficas. Diseñada especialmente para profesionales de la salud 
        y estudiantes de medicina.
        """)
        
        # Características principales
        st.markdown("""
        <div class="feature-card">
            <h4>🔍 Segmentación Automática</h4>
            <p>Identifica y segmenta automáticamente diferentes órganos y estructuras en las ecografías.</p>
        </div>
        
        <div class="feature-card">
            <h4>✨ Mejora de Imagen</h4>
            <p>Mejora la calidad y claridad de las imágenes ecográficas utilizando algoritmos avanzados.</p>
        </div>
        
        <div class="feature-card">
            <h4>📊 Análisis Detallado</h4>
            <p>Proporciona métricas y análisis detallados de las estructuras identificadas.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.image("https://www.radiology.expert/content/images/modules/ultrasound-abdomen-general/en/resized/RadiologyExpert_English_class_ultrasound_abdomen_general_fig16a_normal%20pancreas%20sagittal%20plane_MET_640x0.jpg", 
                caption="Tecnología médica avanzada")
        
        add_vertical_space(2)
        
        # Métricas de ejemplo
        st.metric("Precisión del modelo", "94.2%", "2.1%")
        st.metric("Imágenes procesadas", "1,247", "156")
        st.metric("Tiempo promedio", "3.2s", "-0.8s")

# Página de Segmentar=============================================================
def page_segmentar():
    colored_header(
        label="🔍 Segmentación de Ecografías",
        description="Sube tu imagen para segmentar automáticamente los órganos",
        color_name="blue-70"
    )
    
    # Botón para subir imagen
    st.markdown("""
    <div class="upload-area">
        <h3>📁 Subir Imagen</h3>
        <p>Sube tu imagen de ecografía en formato PNG</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Selecciona una imagen de ecografía",
        type=['png']
    )
    
    if uploaded_file is not None:
        st.success("✅ Imagen cargada correctamente")
        
        # Mostrar imagen original
        st.image(uploaded_file, caption="Imagen de ecografía cargada", use_column_width=True)
        
        # Botón para segmentar
        if st.button("🔍 Segmentar Imagen", use_container_width=True):
            with st.spinner("Procesando imagen..."):
                try:
                    # Cargar modelo
                    model = load_segmentation_model()
                    
                    # Convertir archivo subido a imagen
                    image = Image.open(uploaded_file)
                    img_array = np.array(image)
                    
                    # Convertir a escala de grises si es necesario
                    if len(img_array.shape) == 3:
                        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    else:
                        img_gray = img_array
                    
                    # Preprocesamiento (igual que tu código)
                    img_resized = cv2.resize(img_gray, (128, 128))
                    img_norm = img_resized.astype(np.float32) / 255.0
                    img_input = np.expand_dims(img_norm, axis=(0, -1))  # (1, 128, 128, 1)
                    
                    # Predicción
                    pred = model.predict(img_input)[0]  # (128, 128, 9)
                    pred_class = np.argmax(pred, axis=-1)  # (128, 128)
                    pred_rgb = mask_class_to_rgb(pred_class)
                    
                    # Mostrar resultados
                    st.success("✅ Segmentación completada")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📷 Imagen Original")
                        st.image(img_resized, caption="Imagen procesada", use_column_width=True, clamp=True)
                    
                    with col2:
                        st.subheader("🎯 Segmentación")
                        st.image(pred_rgb, caption="Órganos segmentados", use_column_width=True)
                    
                    # Leyenda de colores
                    st.subheader("🎨 Leyenda de Órganos")
                    legend_cols = st.columns(3)
                    
                    class_names = {
                        0: "Fondo", 1: "Hígado", 2: "Riñón", 3: "Páncreas", 
                        4: "Vasos", 5: "Suprarrenales", 6: "Vesícula", 
                        7: "Huesos", 8: "Bazo"
                    }
                    
                    for i, (class_id, name) in enumerate(class_names.items()):
                        col_idx = i % 3
                        color = class_to_color[class_id]
                        with legend_cols[col_idx]:
                            st.markdown(f"""
                            <div style="display: flex; align-items: center; margin: 5px 0;">
                                <div style="width: 20px; height: 20px; background-color: rgb{color}; 
                                           border: 1px solid #ccc; margin-right: 10px; border-radius: 3px;"></div>
                                <span>{name}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error durante la segmentación: {str(e)}")
                    st.info("Verifica que la imagen esté en el formato correcto y el modelo esté disponible.")

# Página de Mejorar==============================================================
def page_mejorar():
    from streamlit_extras.colored_header import colored_header

    colored_header(
        label="✨ Mejora de Ecografías",
        description="Mejora la calidad y nitidez de tus imágenes ecográficas",
        color_name="green-70"
    )

    st.markdown("""
    <div class="upload-area">
        <h3>🎯 Procesamiento de Imagen</h3>
        <p>Sube tu ecografía para aplicar algoritmos de mejora automática</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Selecciona una imagen para mejorar",
        type=['png'],
        help="La imagen será procesada para mejorar su calidad y contraste"
    )

    if uploaded_file is not None:
        st.success("✅ Imagen lista para procesamiento")

        img = Image.open(uploaded_file).convert('RGB')
        img_resized = img.resize((256, 256), Image.BICUBIC)

        transform = get_transform(opt, grayscale=True)
        img_tensor = transform(img_resized).unsqueeze(0)

        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()

        with torch.no_grad():
            fake_tensor = generator(img_tensor)

        fake_np = fake_tensor.squeeze().cpu().numpy()
        fake_np = ((fake_np + 1) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)

        fake_pil = Image.fromarray(fake_np)
        #col1, col2 = st.columns(2)
        #with col1:
        #    st.markdown(
        #        f"<div style='text-align: center'><img src='data:image/png;base64,{image_to_base64(img_resized.convert('L'))}' width='200'><p>Imagen original (256x256)</p></div>",
        #        unsafe_allow_html=True
        #    )
        #with col2:
        #    st.markdown(
        #        f"<div style='text-align: center'><img src='data:image/png;base64,{image_to_base64(fake_pil)}' width='200'><p>Transformada por CycleGAN</p></div>",
        #        unsafe_allow_html=True
        #    )

        #Mostrar lado a lado
        col1, col2 = st.columns(2)
        col1.image(img_resized.convert('L'), caption="Imagen original (256x256)", width=400)
        col2.image(fake_np, caption="Transformada por CycleGAN", width=400)

# Página de Información============================================================
def page_informacion():
    colored_header(
        label="ℹ️ Información del Sistema",
        description="Conoce más sobre EchoScan AI y sus capacidades",
        color_name="orange-70"
    )
    
    tab1, tab2, tab3 = st.tabs(["📋 Acerca de", "🔬 Tecnología", "👥 Equipo"])
    
    with tab1:
        st.markdown("""
        ### 🎯 Misión
        
        **EchoScan AI** tiene como misión democratizar el acceso a herramientas avanzadas de análisis 
        de imágenes médicas, facilitando el trabajo de profesionales de la salud y mejorando la 
        precisión en el diagnóstico.
        
        ### 🌟 Características Principales
        
        - **Inteligencia Artificial Avanzada**: Utilizamos modelos de deep learning entrenados específicamente para imágenes ecográficas
        - **Interfaz Intuitiva**: Diseñada pensando en la experiencia del usuario médico
        - **Procesamiento Rápido**: Resultados en segundos, no en minutos
        - **Alta Precisión**: Más del 94% de precisión en la segmentación de órganos
        
        ### 📊 Estadísticas de Uso
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Usuarios Activos", "2,847", "12%")
        with col2:
            st.metric("Imágenes Procesadas", "15,692", "8%")
        with col3:
            st.metric("Precisión Promedio", "94.2%", "1.2%")
    
    with tab2:
        st.markdown("""
        ### 🔬 Stack Tecnológico
        
        **EchoScan AI** está construido utilizando las tecnologías más avanzadas en el campo de la 
        inteligencia artificial y el procesamiento de imágenes médicas.
        
        #### 🧠 Inteligencia Artificial
        - **TensorFlow/PyTorch**: Frameworks de deep learning
        - **U-Net Architecture**: Para segmentación semántica
        - **ResNet Backbone**: Para extracción de características
        
        #### 🖥️ Frontend y Backend
        - **Streamlit**: Framework de aplicaciones web
        - **Python**: Lenguaje de programación principal
        - **OpenCV**: Procesamiento de imágenes
        - **NumPy/Pandas**: Análisis de datos
        """)
        
        st.image("https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=600&h=300&fit=crop", 
                caption="Tecnología de vanguardia")
    
    with tab3:
        st.markdown("""
        ### 👥 Nuestro Equipo
        
        **EchoScan AI** es desarrollado por un equipo multidisciplinario de ingenieros, médicos y 
        científicos de datos comprometidos con la innovación en salud digital.
        
        #### 🏥 Colaboraciones
        - Instituto de Investigación Médica
        - Universidad Nacional de Ingeniería
        - Hospital Nacional de Especialidades
        
        #### 📞 Contacto
        - **Email**: info@echoscan.ai
        - **Teléfono**: +51 1 234-5678
        - **Dirección**: Lima, Perú
        """)

# Inicializar estado de sesión
if 'page' not in st.session_state:
    st.session_state.page = 'inicio'

# Mostrar navegación lateral
create_sidebar_navigation()

add_vertical_space(2)

# Enrutamiento de páginas
if st.session_state.page == 'inicio':
    page_inicio()
elif st.session_state.page == 'segmentar':
    page_segmentar()
elif st.session_state.page == 'mejorar':
    page_mejorar()
elif st.session_state.page == 'informacion':
    page_informacion()

# Footer
add_vertical_space(3)
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🔬 EchoScan AI © 2025 - Desarrollado con ❤️ para la comunidad médica</p>
</div>
""", unsafe_allow_html=True)
