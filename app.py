import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import time
import tempfile

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Drone Detection System",
    page_icon=Image.open("favicon.jpg"),
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700;800&family=Inter:wght@400;500&display=swap');
    /* ── Force dark background everywhere ── */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stSidebar"], .main, section[data-testid="stSidebar"] > div {
        background-color: #0a0a0a !important;
        color: #ffffff !important;
    }
    [data-testid="stAppViewContainer"] {
        zoom: 0.9;
    }

    html, body, * {
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Hide Streamlit branding ── */
    #MainMenu, footer { visibility: hidden; }
    [data-testid="stDecoration"] {
        display: none !important;
    }
    header {
        visibility: visible !important;
        background: transparent !important;
        height: 0 !important;
    }

    /* ── Main header ── */
    .main-header {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 4.4rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        letter-spacing: -1px !important;
        margin-bottom: 0 !important;
        line-height: 1.1 !important;
    }
    .main-header span { color: #E8FF00; }
    .sub-header {
        color: #666;
        font-size: 0.95em;
        margin-bottom: 1.5rem;
        letter-spacing: 0.5px;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #111;
        padding: 4px;
        border-radius: 10px;
        border: 1px solid #222;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Grotesk', sans-serif !important;
        border-radius: 8px;
        padding: 8px 24px;
        color: #666 !important;
        background: transparent !important;
        font-weight: 500;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background: #E8FF00 !important;
        color: #000000 !important;
        font-weight: 700 !important;
    }
    /* ── Remove red underline from active tab ── */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: transparent !important;
        height: 0px !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        background-color: transparent !important;
        height: 0px !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        min-width: 320px !important;
        max-width: 320px !important;
        border-right: 1px solid #1a1a1a !important;
        overflow-x: hidden !important;
        overflow-y: hidden !important;
    }
    [data-testid="stSidebar"] > div {
        overflow-x: hidden !important;
        overflow-y: hidden !important;
        padding: 0.5rem 0.9rem 0.8rem 0.9rem !important;
    }
    [data-testid="stSidebarContent"] {
        overflow-x: hidden !important;
        overflow-y: hidden !important;
        padding-top: 0 !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        width: 320px !important;
    }
    section[data-testid="stSidebar"] {
        overflow: hidden !important;
    }
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }
    [data-testid="stSidebarCollapseButton"] span {
        display: none !important;
    }
    button[kind="headerNoPadding"] {
        display: none !important;
    }
    .sidebar-label {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 2px;
        color: #444;
        text-transform: uppercase;
        margin-top: 0 !important;
        margin-bottom: 6px;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: #111 !important;
        border: 1px solid #1e1e1e !important;
        border-radius: 10px !important;
        min-width: 96px !important;
        padding: 10px 12px !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #555 !important;
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.45rem !important;
        font-weight: 700 !important;
    }
/* ── CLEAN STREAMLIT SLIDER FIX ── */

/* Container spacing fix */
[data-testid="stSlider"] {
    padding-right: 6px !important;
}

/* Progress track (yellow line) */
[data-testid="stSlider"] [role="progressbar"] {
    background: #E8FF00 !important;
    height: 3px !important;
    border-radius: 999px !important;
}

/* Slider thumb */
[data-testid="stSlider"] [role="slider"] {
    width: 14px !important;
    height: 14px !important;
    background: #E8FF00 !important;
    border-radius: 50% !important;
    border: none !important;
    box-shadow: 0 0 8px rgba(232, 255, 0, 0.6) !important;
}

/* Remove default gray track */
[data-testid="stSlider"] [data-baseweb="slider"] > div {
    background: transparent !important;
}

# #RESTORE THIS!!!!!!!!1
# /* ── Slider fix ── */
# [data-testid="stSlider"] [data-baseweb="slider"] {
#     padding-right: 0 !important;
#     margin-right: 0 !important;
# }
# [data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {
#     width: 100% !important;
#     right: 0 !important;
# }

# /* ── Slider thumb glow ── */
# [data-testid="stSlider"] input[type="range"]::-webkit-slider-thumb {
#     box-shadow: 0 0 8px 4px rgba(232, 255, 0, 0.6) !important;
# }
# [data-testid="stSlider"] input[type="range"]::-moz-range-thumb {
#     box-shadow: 0 0 8px 4px rgba(232, 255, 0, 0.6) !important;
# }

   #  /* ── Slider track ── */
   # [data-testid="stSlider"] > div > div > div > div {
   #  background: #E8FF00 !important;
   #      }
   #      [data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {
   #  background: #E8FF00 !important;
   #      }
   #  [data-testid="stSlider"] [role="progressbar"] {
   #  background: #E8FF00 !important;
   #  }
   #  [data-testid="stSlider"] div[style*="background"] {
   #  background-color: #E8FF00 !important;
   #  }
   #  [class*="sliderTrack"] {
   #  background: #E8FF00 !important;
   #  }
   #  /* ── Slider thumb → yellow ── */
   #  [data-testid="stSlider"] input[type="range"]::-webkit-slider-thumb {
   #      background: #E8FF00 !important;
   #      border: 2px solid #000 !important;
   #  }
   #  [data-testid="stSlider"] input[type="range"]::-moz-range-thumb {
   #      background: #E8FF00 !important;
   #      border: 2px solid #000 !important;
   #  }
   #  [data-testid="stSlider"] [role="slider"] {
   #      background: #E8FF00 !important;
   #      background-color: #E8FF00 !important;
   #      border: 2px solid #E8FF00 !important;
   #      border-color: #E8FF00 !important;
   #      box-shadow: 0 0 0 4px rgba(232, 255, 0, 0.22) !important;
   #  }
   #  div[data-testid="stSlider"] [role="slider"] {
   #      background-color: #E8FF00 !important;
   #      border-color: #E8FF00 !important;
   #  }

    # /* ── Buttons — default ── */
    # /* Final slider override for deploy: thin yellow line + round thumb only */
    # [data-testid="stSlider"] > div > div > div > div,
    # [data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {
    #     background: transparent !important;
    # }
    # [data-testid="stSlider"] [role="progressbar"] {
    #     height: 2px !important;
    #     background: #E8FF00 !important;
    #     border-radius: 999px !important;
    # }
    # [data-testid="stSlider"] [role="slider"],
    # div[data-testid="stSlider"] [role="slider"] {
    #     width: 14px !important;
    #     height: 14px !important;
    #     background: #E8FF00 !important;
    #     background-color: #E8FF00 !important;
    #     border: 2px solid #E8FF00 !important;
    #     border-color: #E8FF00 !important;
    #     border-radius: 50% !important;
    #     box-shadow: none !important;
    # }


#CHANGED HERE

/* ── Fix sidebar layout overflow (REAL FIX) ── */
[data-testid="stSidebar"] > div {
    width: 100% !important;
}

/* Fix slider container width */
[data-testid="stSlider"] {
    width: 100% !important;
    max-width: 100% !important;
    overflow: hidden !important;
}

/* Prevent internal overflow */
[data-testid="stSlider"] [data-baseweb="slider"] {
    width: 100% !important;
    max-width: 100% !important;
}
#END

    .stButton > button {
        background: #111 !important;
        color: #ffffff !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        border-color: #E8FF00 !important;
        color: #E8FF00 !important;
        background: #E8FF0010 !important;
    }

    /* ── Selected demo button ── */
    .stButton > button[data-selected="true"],
    .stButton > button:focus {
        border-color: #E8FF00 !important;
        color: #E8FF00 !important;
        background: #E8FF0015 !important;
        outline: none !important;
        box-shadow: 0 0 0 2px #E8FF0040 !important;
    }

    /* ── Download button ── */
    [data-testid="stDownloadButton"] > button {
        background: #E8FF00 !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
    }
    [data-testid="stDownloadButton"] > button:hover {
        background: #d4eb00 !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: #111 !important;
        border: 1px dashed #2a2a2a !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #E8FF00 !important;
    }

    /* ── Detection badges ── */
    .detection-badge {
        font-family: 'Space Grotesk', sans-serif !important;
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
        margin: 4px;
        letter-spacing: 0.5px;
    }
    .drone-badge { background: #FF4B4B18; color: #FF4B4B; border: 1px solid #FF4B4B55; }
    .bird-badge  { background: #00C85318; color: #00C853; border: 1px solid #00C85355; }
    .plane-badge { background: #E8FF0018; color: #E8FF00; border: 1px solid #E8FF0055; }
    .kite-badge  { background: #AA00FF18; color: #BB44FF; border: 1px solid #AA00FF55; }

    /* ── Divider ── */
    hr { border-color: #1a1a1a !important; }

    /* ── Progress bar ── */
    [data-testid="stProgressBar"] > div > div {
        background: #E8FF00 !important;
    }

    /* ── Info/warning boxes ── */
    [data-testid="stAlert"] {
        background: #111 !important;
        border: 1px solid #1e1e1e !important;
        border-radius: 8px !important;
        color: #888 !important;
    }
    [data-testid="stAlert"][data-baseweb="notification"] {
        background: #111 !important;
        border: 1px solid #1e1e1e !important;
        border-left: 3px solid #E8FF00 !important;
        border-radius: 8px !important;
        color: #888 !important;
    }
    [data-testid="stAlert"][data-baseweb="notification"] > div {
        background: #111 !important;
    }
    [data-testid="stAlert"][data-baseweb="notification"] * {
        color: #888 !important;
    }

    /* ── Typography ── */
    h1, h2, h3, label {
        font-family: 'Space Grotesk', sans-serif !important;
    }
    h3 { color: #ffffff !important; font-weight: 700 !important; }
    p, li { color: #888 !important; }
    strong { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# ─── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

model = load_model()

# ─── Constants ─────────────────────────────────────────────────
CLASS_COLORS = {
    "drone": (0,   0,   255),
    "bird":  (0,   255, 0  ),
    "plane": (0,   232, 255),
    "kite":  (255, 0,   255),
}
BADGE_CLASS = {
    "drone": "drone-badge",
    "bird":  "bird-badge",
    "plane": "plane-badge",
    "kite":  "kite-badge",
}

# ─── Inference ─────────────────────────────────────────────────
def run_inference(image: Image.Image, conf: float):
    img_array = np.array(image.convert("RGB"))
    img_bgr   = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    start   = time.time()
    results = model(img_bgr, conf=conf, verbose=False)
    elapsed = (time.time() - start) * 1000

    detections = []
    for result in results:
        for box in result.boxes:
            cls_id     = int(box.cls[0])
            cls_name   = model.names[cls_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = CLASS_COLORS.get(cls_name, (255, 255, 255))
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {confidence:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_bgr, (x1, y1 - lh - 8), (x1 + lw, y1), color, -1)
            cv2.putText(img_bgr, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            detections.append({
                "class": cls_name,
                "confidence": round(confidence, 3),
                "bbox": [x1, y1, x2, y2]
            })

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb), detections, elapsed


def process_video(video_path: str, conf: float, progress_bar, status_text):
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        out_path = tmp_file.name
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create MP4 output video.")

    frame_idx      = 0
    all_dets       = []
    preview_frames = []
    preview_step   = max(int(fps // 6), 1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, verbose=False)
        for result in results:
            for box in result.boxes:
                cls_id     = int(box.cls[0])
                cls_name   = model.names[cls_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                color = CLASS_COLORS.get(cls_name, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name} {confidence:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                all_dets.append(cls_name)

        writer.write(frame)

        if frame_idx % preview_step == 0 and len(preview_frames) < 80:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preview_image = Image.fromarray(rgb_frame)
            preview_image.thumbnail((640, 360))
            preview_frames.append(preview_image)

        frame_idx += 1

        if total > 0:
            progress_bar.progress(min(frame_idx / total, 1.0))
            status_text.text(f"Processing frame {frame_idx}/{total}...")

    cap.release()
    writer.release()

    preview_path = None
    if preview_frames:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as tmp_preview:
            preview_path = tmp_preview.name
        preview_frames[0].save(
            preview_path,
            save_all=True,
            append_images=preview_frames[1:],
            duration=max(int(1000 / 6), 1),
            loop=0,
        )

    return out_path, all_dets, preview_path


def show_detections(detections):
    if not detections:
        st.info("No objects detected. Try lowering the confidence threshold.")
        return
    st.markdown("**Detections**")
    badges = ""
    for d in detections:
        badge_cls = BADGE_CLASS.get(d["class"], "drone-badge")
        badges += f'<span class="detection-badge {badge_cls}">{d["class"].upper()} {d["confidence"]}</span>'
    st.markdown(badges, unsafe_allow_html=True)


def format_name(filename: str) -> str:
    return os.path.splitext(filename)[0].replace("_", " ").title()


# ─── Header ────────────────────────────────────────────────────
st.markdown('''
<p class="main-header">DRONE <span>DETECTION</span></p>
<p class="sub-header">CNN-based aerial object detection · YOLOv8s · 12,700+ images · 4 classes</p>
''', unsafe_allow_html=True)
st.markdown("---")

# ─── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-label">Settings</p>', unsafe_allow_html=True)
    conf_threshold = st.slider(
        "Confidence threshold",
        min_value=0.10, max_value=0.95,
        value=0.40, step=0.05,
        help="Lower = more detections | Higher = fewer, more confident"
    )
    st.markdown("---")
    st.markdown('<p class="sidebar-label">Model</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.metric("mAP50",   "0.820")
    col2.metric("Speed",   "~3ms")
    col1.metric("Classes", "4")
    col2.metric("Params",  "11.1M")
    st.markdown("---")
    st.markdown('<p class="sidebar-label">Classes</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;flex-direction:column;gap:8px;margin-top:4px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:10px;height:10px;border-radius:2px;background:#FF4B4B;flex-shrink:0;"></div>
            <span style="color:#888;font-size:0.85rem;">Drone</span>
        </div>
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:10px;height:10px;border-radius:2px;background:#00C853;flex-shrink:0;"></div>
            <span style="color:#888;font-size:0.85rem;">Bird</span>
        </div>
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:10px;height:10px;border-radius:2px;background:#E8FF00;flex-shrink:0;"></div>
            <span style="color:#888;font-size:0.85rem;">Plane</span>
        </div>
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:10px;height:10px;border-radius:2px;background:#BB44FF;flex-shrink:0;"></div>
            <span style="color:#888;font-size:0.85rem;">Kite</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p class="sidebar-label">Training Stats</p>', unsafe_allow_html=True)
    st.markdown("""
    Model: YOLOv8s  
    Epochs: 50  
    Image size: 512px  
    Split: 80 / 10 / 10  
    """)

# ─── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["  Upload Image  ", "  Upload Video  ", "  Demo Examples  "])

# ══ Tab 1: Image Upload ════════════════════════════════════════
with tab1:
    st.subheader("Upload an image")
    uploaded = st.file_uploader(
        "Drag and drop or browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded:
        image = Image.open(uploaded)
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.markdown("**Original**")
            st.image(image, use_container_width=True)
        with col2:
            st.markdown("**Detection Result**")
            with st.spinner("Running inference..."):
                result_img, detections, elapsed = run_inference(image, conf_threshold)
            st.image(result_img, use_container_width=True)

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total detections", len(detections))
        m2.metric("Drones found",     sum(1 for d in detections if d["class"] == "drone"))
        m3.metric("Inference time",   f"{elapsed:.1f}ms")
        m4.metric("Confidence",       conf_threshold)
        st.markdown("---")
        show_detections(detections)

# ══ Tab 2: Video Upload ════════════════════════════════════════
with tab2:
    st.subheader("Upload a video")
    st.caption("Model processes each frame and returns an annotated GIF preview + downloadable MP4.")

    uploaded_video = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed"
    )

    demo_video_dir  = "demo_videos"
    demo_video_path = None
    if os.path.exists(demo_video_dir):
        demo_videos = [f for f in os.listdir(demo_video_dir)
                       if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        if demo_videos:
            st.markdown("**Or try the demo video:**")
            use_demo = st.session_state.get("use_demo_video", False)
            btn_style = "border: 1px solid #E8FF00 !important; color: #E8FF00 !important; background: #E8FF0015 !important;" if use_demo else ""
            if st.button("Bird vs Drone — Real World Footage", use_container_width=True):
                st.session_state["use_demo_video"] = True
                st.rerun()
            if st.session_state.get("use_demo_video"):
                demo_video_path = os.path.join(demo_video_dir, demo_videos[0])

    video_source = None
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_source = tfile.name
        st.session_state["use_demo_video"] = False
    elif demo_video_path:
        video_source = demo_video_path

    if video_source:
        st.markdown("---")
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.markdown("**Original Video**")
            st.video(video_source)

        with col2:
            st.markdown("**Processing**")
            progress_bar = st.progress(0)
            status_text  = st.empty()

            with st.spinner("Running inference on all frames..."):
                out_path, all_dets, preview_path = process_video(
                    video_source, conf_threshold, progress_bar, status_text
                )

            status_text.text("Done!")
            st.markdown("**Annotated Preview**")
            if preview_path:
                st.image(preview_path, use_container_width=True)
            else:
                st.info("Preview unavailable — download the annotated video below.")

            with open(out_path, "rb") as f:
                data = f.read()
            st.download_button(
                "Download annotated video",
                data,
                file_name="drone_detection_result.mp4",
                mime="video/mp4",
                use_container_width=True
            )

        st.markdown("---")
        if all_dets:
            from collections import Counter
            counts = Counter(all_dets)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total detections", len(all_dets))
            m2.metric("Drone detections", counts.get("drone", 0))
            m3.metric("Bird detections",  counts.get("bird",  0))
            m4.metric("Other detections", counts.get("plane", 0) + counts.get("kite", 0))

# ══ Tab 3: Demo Examples ═══════════════════════════════════════
with tab3:
    st.subheader("Preloaded examples")
    st.caption("Select an example to see the model in action on real-world images.")

    demo_dir = "demo_images"
    if not os.path.exists(demo_dir):
        st.warning("demo_images/ folder not found.")
    else:
        demo_files = [f for f in os.listdir(demo_dir)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not demo_files:
            st.warning("No images found in demo_images/.")
        else:
            if "selected_demo" not in st.session_state:
                st.session_state["selected_demo"] = demo_files[0]

            cols = st.columns(min(len(demo_files), 3))
            for i, fname in enumerate(demo_files):
                with cols[i % 3]:
                    is_selected = st.session_state["selected_demo"] == fname
                    label = f"✦ {format_name(fname)}" if is_selected else format_name(fname)
                    if st.button(label, key=f"demo_{fname}", use_container_width=True):
                        st.session_state["selected_demo"] = fname
                        st.rerun()

            st.markdown("---")
            selected = st.session_state["selected_demo"]
            image    = Image.open(os.path.join(demo_dir, selected))

            col1, col2 = st.columns(2, gap="medium")
            with col1:
                st.markdown(f"**Original — {format_name(selected)}**")
                st.image(image, use_container_width=True)
            with col2:
                st.markdown("**Detection Result**")
                with st.spinner("Running inference..."):
                    result_img, detections, elapsed = run_inference(image, conf_threshold)
                st.image(result_img, use_container_width=True)

            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total detections", len(detections))
            m2.metric("Drones found",     sum(1 for d in detections if d["class"] == "drone"))
            m3.metric("Inference time",   f"{elapsed:.1f}ms")
            m4.metric("Confidence",       conf_threshold)
            st.markdown("---")
            show_detections(detections)
