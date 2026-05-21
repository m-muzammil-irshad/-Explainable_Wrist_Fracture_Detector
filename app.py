import os
import cv2
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.saving import register_keras_serializable
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# 1. CUSTOM CLASSES
# ─────────────────────────────────────────────────────────────

@register_keras_serializable()
class SpatialAttention(keras.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(
            1, kernel_size, padding="same",
            activation="sigmoid", use_bias=False,
        )

    def call(self, x):
        avg    = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_   = tf.reduce_max(x,  axis=-1, keepdims=True)
        concat = tf.concat([avg, max_], axis=-1)
        attn   = self.conv(concat)
        return x * attn, attn

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"kernel_size": self.kernel_size})
        return cfg


@register_keras_serializable()
def focal_loss(gamma=2.0, alpha=0.60):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce    = keras.losses.binary_crossentropy(y_true, y_pred)
        p_t    = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return alpha * tf.pow(1 - p_t, gamma) * bce
    return loss


# ─────────────────────────────────────────────────────────────
# 2. LOAD MODEL
# ─────────────────────────────────────────────────────────────

model = keras.models.load_model(
    "best_wrist_model.keras",
    custom_objects={
        "SpatialAttention": SpatialAttention,
        "loss": focal_loss(gamma=2, alpha=0.60),
    },
)
print("✅ Model loaded! Input shape:", model.input_shape)
print("Model layers:", [l.name for l in model.layers])

# ── Warmup: compile graph once at startup so first user request is fast ──
_dummy = np.zeros((1, 320, 320, 3), dtype=np.float32)
model.predict(_dummy, verbose=0)
print("✅ Warmup done — graph compiled")

# ── Find DenseNet sub-model once at startup ──
DENSENET = None
DENSENET_IDX = 0
for _i, _layer in enumerate(model.layers):
    if hasattr(_layer, "layers") and len(getattr(_layer, "layers", [])) > 10:
        DENSENET = _layer
        DENSENET_IDX = _i
        break

TARGET_CONV = "conv5_block16_2_conv"
TARGET_LAYER = DENSENET.get_layer(TARGET_CONV)
TAIL_LAYERS  = model.layers[DENSENET_IDX + 1:]   # layers after densenet
print(f"✅ Grad-CAM target: {TARGET_CONV}")


# ─────────────────────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────────────────────

def preprocess_xray(img_rgb):
    if img_rgb.max() <= 1.0:
        img_uint8 = (img_rgb * 255).astype(np.uint8)
    else:
        img_uint8 = img_rgb.astype(np.uint8)

    img_gray  = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)
    img_blur  = cv2.GaussianBlur(img_clahe, (0, 0), 3)
    img_sharp = cv2.addWeighted(img_clahe, 1.5, img_blur, -0.5, 0)

    sobelx = cv2.Sobel(img_sharp, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_sharp, cv2.CV_64F, 0, 1, ksize=3)
    edges  = np.sqrt(sobelx**2 + sobely**2)
    edges  = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    ch1 = img_sharp.astype("float32") / 255.0
    ch2 = img_clahe.astype("float32") / 255.0
    ch3 = edges.astype("float32")     / 255.0

    processed = np.stack([ch1, ch2, ch3], axis=-1)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    return (processed - mean) / std


# ─────────────────────────────────────────────────────────────
# 4. GRAD-CAM  (robust — searches all nested layers)
# ─────────────────────────────────────────────────────────────

def make_gradcam(img_array):
    """Fast GradCAM: single forward pass with hook, cached globals."""
    inp = tf.cast(img_array, tf.float32)
    captured = {}
    original_call = TARGET_LAYER.call

    def hooked_call(x, **kwargs):
        out = original_call(x, **kwargs)
        captured["out"] = out
        return out

    TARGET_LAYER.call = hooked_call
    try:
        with tf.GradientTape() as tape:
            preds = model(inp, training=False)
            if "out" not in captured:
                raise ValueError("Target conv layer was not called")
            conv_out = captured["out"]
            tape.watch(conv_out)
            loss = tf.reshape(preds, [-1])[0]
        grads = tape.gradient(loss, conv_out)
    finally:
        TARGET_LAYER.call = original_call

    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()





def overlay_heatmap(orig, heatmap, alpha=0.45):
    h       = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    h       = np.uint8(255 * h)
    colored = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    out     = orig.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────
# 5. PLOT HELPERS  (white background — fixes black box issue)
# ─────────────────────────────────────────────────────────────

def make_confidence_plot(prob_frac, prob_norm):
    fig, ax = plt.subplots(figsize=(5, 1.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.barh(
        ["Normal", "Fracture"],
        [prob_norm * 100, prob_frac * 100],
        color=["#27ae60", "#e74c3c"],
        height=0.5, edgecolor="none",
    )
    for bar, val in zip(bars, [prob_norm * 100, prob_frac * 100]):
        ax.text(
            min(val + 1, 95), bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", color="#222",
            fontsize=12, fontweight="bold",
        )
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)", color="#444", fontsize=10)
    ax.tick_params(colors="#444", labelsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False)
    plt.tight_layout(pad=0.6)
    return fig


def make_gradcam_plot(img_resized, heatmap):
    overlay = overlay_heatmap(img_resized, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    fig.patch.set_facecolor("white")

    titles = ["Original X-ray", "Grad-CAM Heatmap", "Overlay (Focus Area)"]
    imgs   = [
        img_resized,
        cv2.cvtColor(
            cv2.applyColorMap(
                np.uint8(255 * cv2.resize(heatmap, (320, 320))),
                cv2.COLORMAP_JET,
            ), cv2.COLOR_BGR2RGB),
        overlay,
    ]
    cmaps = ["gray", None, None]

    for ax, title, im, cmap in zip(axes, titles, imgs, cmaps):
        ax.set_facecolor("white")
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, color="#222", fontsize=11, pad=8, fontweight="bold")
        ax.axis("off")

    sm   = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Activation intensity", fontsize=9)

    fig.suptitle(
        "Grad-CAM: Areas the model focused on",
        color="#333", fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout(pad=1.0)
    return fig


def make_gradcam_error_plot(msg):
    fig, ax = plt.subplots(figsize=(7, 2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fff8f8")
    ax.text(0.5, 0.5, f"⚠ Grad-CAM unavailable\n{msg}",
            ha="center", va="center", color="#c0392b",
            transform=ax.transAxes, fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# 6. MAIN PREDICT
# ─────────────────────────────────────────────────────────────

def predict(image):
    if image is None:
        return "Please upload an X-ray image.", None, None

    img_resized = cv2.resize(image, (320, 320))
    processed   = preprocess_xray(img_resized)
    inp         = np.expand_dims(processed, 0).astype(np.float32)

    prob_frac = float(model(inp, training=False).numpy()[0][0])
    prob_norm = 1.0 - prob_frac
    is_frac   = prob_frac >= 0.5
    conf      = prob_frac if is_frac else prob_norm

    label = (
        f"🦴 FRACTURE DETECTED  ({conf*100:.1f}% confidence)"
        if is_frac
        else f"✅ NORMAL — No Fracture  ({conf*100:.1f}% confidence)"
    )

    fig_conf = make_confidence_plot(prob_frac, prob_norm)

    try:
        heatmap = make_gradcam(inp)
        fig_cam = make_gradcam_plot(img_resized, heatmap)
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        fig_cam = make_gradcam_error_plot(str(e))

    plt.close("all")   # free memory after each request
    return label, fig_conf, fig_cam


# ─────────────────────────────────────────────────────────────
# 7. GRADIO UI
# ─────────────────────────────────────────────────────────────

HEADER = """
<div style="
  background: linear-gradient(135deg, #1a237e 0%, #1565c0 50%, #0288d1 100%);
  border-radius: 16px;
  padding: 28px 24px 22px;
  text-align: center;
  margin-bottom: 8px;
  box-shadow: 0 4px 24px rgba(26,35,126,0.18);
">
<div style="font-size: 2.2rem; margin-bottom: 6px; color: white;">&#x1F9B4; &#x2665;</div>
  <h1 style="
    color: white; font-size: 1.9rem; font-weight: 700;
    margin: 0 0 8px; letter-spacing: -0.5px;
  ">Wrist Fracture Detector</h1>
  <p style="
    color: rgba(255,255,255,0.85); font-size: 0.92rem;
    max-width: 540px; margin: auto; line-height: 1.65;
  ">
    AI-powered analysis using <strong>DenseNet121 + CBAM Attention</strong>.
    Upload a wrist X-ray to detect fractures and visualize model attention with
    <strong>Grad-CAM</strong>.
  </p>
</div>
"""

TIPS = """
<div style="
  background: #f0f7ff;
  border-left: 4px solid #1565c0;
  border-radius: 8px;
  padding: 12px 16px;
  margin-top: 10px;
  font-size: 0.88rem;
  color: #1a237e;
">
  <strong>📋 Tips for best results</strong><br>
  ✔ Clear, high-resolution wrist X-ray<br>
  ✔ PA (front) or lateral (side) view<br>
  ✔ Wrist centered and fully visible<br>
  ✘ Avoid blurry or heavily cropped images
</div>
"""

DISCLAIMER = """
<div style="
  background: #fff8e1;
  border-left: 4px solid #f9a825;
  border-radius: 8px;
  padding: 10px 16px;
  margin-top: 12px;
  font-size: 0.85rem;
  color: #5f3500;
">
  ⚠️ <strong>Medical Disclaimer:</strong> This tool is for <strong>research and
  educational purposes only</strong>. It is <em>not</em> a substitute for professional
  medical diagnosis. Always consult a qualified radiologist or physician.
</div>
"""

css = """
.result-box textarea {
  font-size: 1.15rem !important;
  font-weight: 600 !important;
  text-align: center !important;
  border-radius: 10px !important;
  padding: 14px !important;
  border: 2px solid #1565c0 !important;
  background: #f0f7ff !important;
  color: #1a237e !important;
}
.section-heading {
  font-size: 1rem !important;
  font-weight: 600 !important;
  color: #1565c0 !important;
  margin: 14px 0 4px !important;
  padding-left: 2px;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    title="Wrist Fracture Detector",
    css=css,
) as demo:

    gr.HTML(HEADER)

    with gr.Row(equal_height=False):
        # ── LEFT COLUMN ──────────────────────────────────────
        with gr.Column(scale=1, min_width=280):
            image_input = gr.Image(
                type="numpy",
                label="📂 Upload Wrist X-ray",
                height=300,
            )
            submit_btn = gr.Button(
                "🔍  Analyze X-ray",
                variant="primary",
                size="lg",
            )
            gr.HTML(TIPS)

        # ── RIGHT COLUMN ─────────────────────────────────────
        with gr.Column(scale=2):
            gr.HTML('<p class="section-heading">🔬 Diagnosis Result</p>')
            result_label = gr.Textbox(
                label="",
                interactive=False,
                lines=1,
                text_align="center",
                elem_classes=["result-box"],
            )
            gr.HTML('<p class="section-heading">📊 Probability Breakdown</p>')
            confidence_plot = gr.Plot(label="", container=False)
            gr.HTML('<p class="section-heading">🧠 Grad-CAM — Where the model looked</p>')
            gradcam_plot = gr.Plot(label="", container=False)

    gr.HTML(DISCLAIMER)

    submit_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[result_label, confidence_plot, gradcam_plot],
    )

demo.launch()
