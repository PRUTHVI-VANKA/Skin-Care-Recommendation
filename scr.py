{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "ce0c51b7-7a95-431d-a342-149b314824a9",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-04-17 13:53:17.427 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-17 13:53:17.427 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-17 13:53:17.428 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-17 13:53:17.430 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-17 13:53:17.711 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-17 13:53:17.713 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-17 13:53:17.713 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-17 13:53:17.714 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-17 13:53:17.715 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-17 13:53:17.715 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-17 13:53:17.716 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-17 13:53:17.717 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-17 13:53:17.717 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-17 13:53:17.718 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-17 13:53:17.718 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-17 13:53:17.719 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "from PIL import Image\n",
    "\n",
    "# 1. Load your model\n",
    "st.sidebar.write(\"🔄 Loading model...\")\n",
    "model = tf.keras.models.load_model(\"pv.keras\")\n",
    "\n",
    "# 2. Define class names\n",
    "skin_type_classes = ['Dry', 'Normal', 'Oily']\n",
    "acne_classes     = ['Low', 'Moderate', 'Severe']\n",
    "\n",
    "# 3. Image preprocessing helper\n",
    "def preprocess_image(img: Image.Image, size=(224, 224)):\n",
    "    img = img.resize(size)\n",
    "    arr = np.array(img) / 255.0\n",
    "    return np.expand_dims(arr, axis=0)  # shape (1,224,224,3)\n",
    "\n",
    "# 4. Streamlit layout\n",
    "st.set_page_config(page_title=\"Skin & Acne Predictor\", layout=\"centered\")\n",
    "st.title(\"🌟 Skin Type & Acne Severity Predictor\")\n",
    "st.write(\"Upload a face photo; the model will tell you your skin type **and** acne severity.\")\n",
    "\n",
    "uploaded_file = st.file_uploader(\"📤 Choose an image...\", type=[\"jpg\",\"jpeg\",\"png\"])\n",
    "if uploaded_file:\n",
    "    img = Image.open(uploaded_file).convert(\"RGB\")\n",
    "    st.image(img, caption=\"Uploaded Image\", use_column_width=True)\n",
    "    \n",
    "    # 5. Run prediction\n",
    "    x = preprocess_image(img)\n",
    "    preds = model.predict(x)\n",
    "    \n",
    "    # 6. Split outputs\n",
    "    # If your model has two output heads:\n",
    "    if isinstance(preds, list) and len(preds) == 2:\n",
    "        skin_pred, acne_pred = preds\n",
    "    else:\n",
    "        # Single array of length 6 → first 3 = skin, last 3 = acne\n",
    "        skin_pred, acne_pred = np.split(preds, 2, axis=1)\n",
    "    \n",
    "    # 7. Decode results\n",
    "    skin_idx = np.argmax(skin_pred[0])\n",
    "    acne_idx = np.argmax(acne_pred[0])\n",
    "    skin_label = skin_type_classes[skin_idx]\n",
    "    acne_label = acne_classes[acne_idx]\n",
    "    skin_conf  = skin_pred[0][skin_idx] * 100\n",
    "    acne_conf  = acne_pred[0][acne_idx] * 100\n",
    "    \n",
    "    # 8. Display\n",
    "    st.subheader(\"🧴 Skin Type\")\n",
    "    st.success(f\"{skin_label} ({skin_conf:.1f}%)\")\n",
    "    st.subheader(\"💥 Acne Severity\")\n",
    "    st.success(f\"{acne_label} ({acne_conf:.1f}%)\")\n",
    "    st.balloons()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "95dff1e6-dc24-4e74-9987-163d904f6fe0",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.16"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
