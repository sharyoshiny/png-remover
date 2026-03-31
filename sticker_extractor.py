import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io
import zipfile

st.set_page_config(page_title="Sticker Extractor", page_icon="✂️", layout="wide")

st.title("✂️ Sticker Sheet Extractor")
st.caption("Upload a sticker sheet → auto-detects each sticker → exports as individual PNGs with transparent background")

# --- Sidebar settings ---
with st.sidebar:
    st.header("Detection Settings")
    min_area = st.slider("Min sticker size (px²)", 1000, 30000, 8000, 500,
                         help="Increase if small noise is detected as stickers")
    padding = st.slider("Padding around each sticker (px)", 0, 50, 15)
    dilation = st.slider("Merge nearby regions (px)", 5, 60, 25,
                         help="Increase to merge stickers that appear connected")
    bg_threshold = st.slider("Background brightness threshold", 180, 255, 230,
                              help="Pixels brighter than this are treated as background")

# --- Image upload ---
uploaded = st.file_uploader("Upload your sticker sheet", type=["png", "jpg", "jpeg", "webp"])

if not uploaded:
    st.info("Upload a sticker sheet image to get started.")
    st.stop()

# Load image
file_bytes = np.frombuffer(uploaded.read(), np.uint8)
img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(img_rgb)

col_preview, col_result = st.columns([1, 2])

with col_preview:
    st.subheader("Uploaded sheet")
    st.image(pil_img, use_container_width=True)

# --- Detect stickers ---
def find_sticker_boxes(image_rgb, threshold, dilation_px, min_area_px):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    # Pixels darker than threshold → part of a sticker
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((dilation_px, dilation_px), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) >= min_area_px:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, w, h))
    # Sort top-to-bottom, left-to-right
    boxes.sort(key=lambda b: (b[1] // 120, b[0]))
    return boxes

boxes = find_sticker_boxes(img_rgb, bg_threshold, dilation, min_area)

# Draw bounding boxes for preview
preview = img_rgb.copy()
for i, (x, y, w, h) in enumerate(boxes):
    cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 80, 80), 3)
    cv2.putText(preview, str(i + 1), (x + 4, y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 80, 80), 2)

with col_preview:
    st.subheader(f"Detected: {len(boxes)} stickers")
    st.image(preview, use_container_width=True)

if len(boxes) == 0:
    st.warning("No stickers detected. Try lowering the Min sticker size or adjusting the threshold.")
    st.stop()

# --- Extract & remove backgrounds ---
def crop_sticker(image_rgb, bbox, pad):
    x, y, w, h = bbox
    h_img, w_img = image_rgb.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_img, x + w + pad)
    y2 = min(h_img, y + h + pad)
    return image_rgb[y1:y2, x1:x2]

def remove_bg(crop_rgb):
    pil = Image.fromarray(crop_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    result_bytes = remove(buf.getvalue())
    return Image.open(io.BytesIO(result_bytes)).convert("RGBA")

with col_result:
    st.subheader("Extracted stickers (transparent background)")

    if st.button("Extract all stickers", type="primary"):
        progress = st.progress(0, text="Removing backgrounds…")
        extracted = []

        cols = st.columns(4)

        for i, bbox in enumerate(boxes):
            crop = crop_sticker(img_rgb, bbox, padding)
            try:
                result_pil = remove_bg(crop)
            except Exception as e:
                st.error(f"Sticker {i+1} failed: {e}")
                continue

            extracted.append((f"sticker_{i+1:02d}.png", result_pil))
            progress.progress((i + 1) / len(boxes), text=f"Processing {i+1}/{len(boxes)}…")

            # Show in grid
            buf = io.BytesIO()
            result_pil.save(buf, format="PNG")
            cols[i % 4].image(result_pil, caption=f"#{i+1}", use_container_width=True)

        progress.empty()

        if extracted:
            # Build zip
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname, img in extracted:
                    img_buf = io.BytesIO()
                    img.save(img_buf, format="PNG")
                    zf.writestr(fname, img_buf.getvalue())
            zip_buf.seek(0)

            st.success(f"Done! {len(extracted)} stickers extracted.")
            st.download_button(
                label="⬇️ Download all as ZIP",
                data=zip_buf,
                file_name="stickers.zip",
                mime="application/zip",
            )
