
import os
import subprocess
import sys

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from model import Generator
import torch


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Fashion DCGAN Generator",
    page_icon="👕",
    layout="wide"
)

st.title("👕 Fashion Image Generator using DCGAN")
st.write(
    "Train a Deep Convolutional GAN on Fashion-MNIST "
    "and generate new clothing-like images."
)

OUTPUT_DIR = "outputs"
GENERATED_DIR = os.path.join(OUTPUT_DIR, "generated")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOSS_FILE = os.path.join(OUTPUT_DIR, "losses.csv")

os.makedirs(GENERATED_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# -----------------------------
# Sidebar controls
# -----------------------------

st.sidebar.header("Training Controls")

epochs = st.sidebar.slider(
    "Number of epochs", min_value=1, max_value=30, value=5
)

dataset_size = st.sidebar.selectbox(
    "Dataset size", [1000, 2000, 3000, 5000], index=1
)

batch_size = st.sidebar.selectbox(
    "Batch size", [32, 64, 128], index=1
)

learning_rate = st.sidebar.number_input(
    "Learning rate",
    min_value=0.00001,
    max_value=0.01,
    value=0.0002,
    format="%.5f"
)

st.sidebar.caption("Training runs on CPU or a compatible CUDA GPU.")


# -----------------------------
# Summary metrics
# -----------------------------

loss_df = pd.DataFrame()

if os.path.exists(LOSS_FILE):
    try:
        loss_df = pd.read_csv(LOSS_FILE)
    except pd.errors.EmptyDataError:
        pass

latest_g = (
    float(loss_df["generator_loss"].iloc[-1])
    if not loss_df.empty else 0.0
)

latest_d = (
    float(loss_df["discriminator_loss"].iloc[-1])
    if not loss_df.empty else 0.0
)

checkpoints = [
    f for f in os.listdir(CHECKPOINT_DIR)
    if f.endswith(".pth")
]

current_epoch = (
    int(loss_df["epoch"].iloc[-1])
    if not loss_df.empty else 0
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Latest Epoch", current_epoch)
c2.metric("Training Images", dataset_size)
c3.metric("Generator Loss", f"{latest_g:.4f}")
c4.metric("Discriminator Loss", f"{latest_d:.4f}")


# -----------------------------
# Start training
# -----------------------------

st.header("⚙️ Train your model")

if st.button("🚀 Start Training", type="primary"):
    st.info("Training has started. Watch the terminal for progress.")

    env = os.environ.copy()
    env["DCGAN_EPOCHS"] = str(epochs)
    env["DCGAN_DATASET_SIZE"] = str(dataset_size)
    env["DCGAN_BATCH_SIZE"] = str(batch_size)
    env["DCGAN_LEARNING_RATE"] = str(learning_rate)

    try:
        result = subprocess.run(
            [sys.executable, "train.py"],
            env=env,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            st.success("Training completed successfully!")
            st.code(result.stdout[-5000:] or "Training completed.")
            st.rerun()
        else:
            st.error("Training failed. Check the error below.")
            st.code(result.stderr[-5000:])

    except Exception as error:
        st.error(f"Could not start training: {error}")


# -----------------------------
# Loss charts
# -----------------------------

st.header("📉 Training Losses")

if not loss_df.empty:
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        loss_df["epoch"],
        loss_df["generator_loss"],
        marker="o",
        label="Generator Loss"
    )

    ax.plot(
        loss_df["epoch"],
        loss_df["discriminator_loss"],
        marker="o",
        label="Discriminator Loss"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Generator vs Discriminator Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(loss_df, use_container_width=True)

    st.download_button(
        "Download Loss CSV",
        data=loss_df.to_csv(index=False),
        file_name="losses.csv",
        mime="text/csv"
    )
else:
    st.info("Train the model to see the loss charts.")


# -----------------------------
# Generated image gallery
# -----------------------------

st.header("🖼️ Generated Clothing Images")

image_files = sorted(
    [
        f for f in os.listdir(GENERATED_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
)

if image_files:
    columns = st.columns(4)

    for i, filename in enumerate(image_files):
        image_path = os.path.join(GENERATED_DIR, filename)

        with columns[i % 4]:
            st.image(
                Image.open(image_path),
                caption=filename,
                use_container_width=True
            )
else:
    st.info("Generated images will appear here after training.")


# -----------------------------
# Checkpoints
# -----------------------------

st.header("💾 Saved Checkpoints")

if checkpoints:
    for filename in sorted(checkpoints):
        checkpoint_path = os.path.join(CHECKPOINT_DIR, filename)

        with open(checkpoint_path, "rb") as file:
            st.download_button(
                label=f"Download {filename}",
                data=file.read(),
                file_name=filename,
                key=filename
            )
else:
    st.info("No checkpoints found yet.")

st.caption(
    "Note: DCGAN-generated images may be blurry or unclear after only "
    "a few epochs. This is normal for an early experiment."
)
