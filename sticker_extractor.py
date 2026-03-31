import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile

st.set_page_config(page_title="Sticker Extractor", page_icon="✂️", layout="wide")

st.title("✂️ Sticker Sheet Extractor")
st.caption("Upload a sticker sheet → auto-detects each sticker → exports as individual PNGs with transparent background")

# --- Sidebar settings ---
with st.sidebar:
    st.header("⚙️ Detection Settings")
    min_area    = st.slider("Min sticker size (px²)",        1000,  30000, 8000, 500)
    padding     = st.slider("Padding around sticker (px)",      0,     60,   15)
    dilation    = st.slider("Merge nearby regions (px)",        5,     80,   25)
    bg_thresh   = st.slider("Background brightness threshold", 180,   255,  230)
    tolerance   = st.slider("Background removal tolerance",      5,    80,   30,
                            help="Higher = removes more of the background color")

# --- Upload ---
uploaded = st.file_uploader("Upload your sticker sheet", type=["png","jpg","jpeg","webp"])
if not uploaded:
    st.info("⬆️ Upload a sticker sheet image to get started.")
    st.stop()

# Load
file_bytes = np.frombuffer(uploaded.read(), np.uint8)
img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Uploaded sheet")
    st.image(img_rgb, use_container_width=True)

# --- Detect bounding boxes ---
def find_sticker_boxes(img, threshold, dil_px, min_area_px):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    kernel  = np.ones((dil_px, dil_px), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) >= min_area_px:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: (b[1] // 120, b[0]))
    return boxes

boxes = find_sticker_boxes(img_rgb, bg_thresh, dilation, min_area)

# Draw preview
preview = img_rgb.copy()
for i, (x, y, w, h) in enumerate(boxes):
    cv2.rectangle(preview, (x, y), (x+w, y+h), (255, 60, 60), 3)
    cv2.putText(preview, str(i+1), (x+5, y+30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 60, 60), 2)

with col_left:
    st.subheader(f"Detected: {len(boxes)} stickers")
    st.image(preview, use_container_width=True)

if len(boxes) == 0:
    st.warning("No stickers found. Try lowering Min sticker size or the threshold.")
    st.stop()

# --- Background removal (flood-fill from corners) ---
def remove_background(crop_rgb, tol):
    """
    Flood-fill from all 4 corners to find the background colour,
    then make every pixel within `tol` distance of that colour transparent.
    Works great for sticker sheets with uniform cream/white backgrounds.
    """
    h, w = crop_rgb.shape[:2]
    rgba  = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2RGBA)
    pil   = Image.fromarray(rgba)

    # Sample background from the 4 corners (most common colour)
    corners = [
        crop_rgb[0,   0  ],
        crop_rgb[0,   w-1],
        crop_rgb[h-1, 0  ],
        crop_rgb[h-1, w-1],
    ]
    bg_color = np.mean(corners, axis=0).astype(np.uint8)

    # Build alpha mask: pixels close to bg_color → transparent
    diff  = np.abs(crop_rgb.astype(np.int16) - bg_color.astype(np.int16))
    dist  = np.max(diff, axis=2)          # max channel distance
    alpha = np.where(dist <= tol, 0, 255).astype(np.uint8)

    # Flood fill from corners to only remove connected background
    mask_flood = np.zeros((h+2, w+2), np.uint8)
    seed_pts   = [(0,0),(0,w-1),(h-1,0),(h-1,w-1)]
    alpha_copy = alpha.copy()
    for sp in seed_pts:
        cv2.floodFill(alpha_copy, mask_flood, (sp[1], sp[0]), 0,
                      loDiff=(tol,), upDiff=(tol,))

    # Combine: only remove bg where flood fill reached
    final_alpha = np.where(alpha_copy == 0, 0, 255).astype(np.uint8)

    # Put back into RGBA
    rgba_out = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2RGBA)
    rgba_out[:, :, 3] = final_alpha
    return Image.fromarray(rgba_out)


def crop_sticker(img, bbox, pad):
    x, y, w, h = bbox
    H, W = img.shape[:2]
    x1, y1 = max(0, x-pad), max(0, y-pad)
    x2, y2 = min(W, x+w+pad), min(H, y+h+pad)
    return img[y1:y2, x1:x2]


# --- Extract button ---
with col_right:
    st.subheader("Extracted stickers")

    if st.button("✂️ Extract all stickers", type="primary"):
        progress  = st.progress(0, text="Extracting…")
        extracted = []
        grid_cols = st.columns(4)

        for i, bbox in enumerate(boxes):
            crop   = crop_sticker(img_rgb, bbox, padding)
            result = remove_background(crop, tolerance)
            extracted.append((f"sticker_{i+1:02d}.png", result))
            progress.progress((i+1)/len(boxes), text=f"Processing {i+1}/{len(boxes)}…")

            buf = io.BytesIO()
            result.save(buf, format="PNG")
            grid_cols[i % 4].image(result, caption=f"#{i+1}", use_container_width=True)

        progress.empty()

        # Build zip
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname, img in extracted:
                img_buf = io.BytesIO()
                img.save(img_buf, format="PNG")
                zf.writestr(fname, img_buf.getvalue())
        zip_buf.seek(0)

        st.success(f"✅ Done! {len(extracted)} stickers extracted.")
        st.download_button(
            label="⬇️ Download all as ZIP",
            data=zip_buf,
            file_name="stickers.zip",
            mime="application/zip",
        )
