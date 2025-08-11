# 🖐️ Virtual Gesture Controller

An interactive **hand gesture-based control interface** built using **Streamlit**, **MediaPipe**, and **OpenCV**. This app lets you control functions like **mouse clicks**, **volume**, **brightness**, and even take **screenshots**—all using simple hand gestures in front of your webcam.

[**Try the live app here!**](https://virtual-mouse-app.streamlit.app/)
---

## 🚀 Features

- 🖱️ **Mouse Controls**
  - **Left Click:** Index down + Ring & Pinky up
  - **Right Click:** Middle down + Ring & Pinky up
  - **Double Click:** Index & Middle down + Ring & Pinky up

- 📸 **Screenshot Capture**
  - Make a closed fist (all fingers down)

- 🔊 **Volume Control**
  - Raise Index + Middle + Ring, then pinch (Index + Thumb)

- 💡 **Brightness Control**
  - Raise Index + Middle + Pinky, then pinch (Index + Thumb)

---

## 📷 How It Works

- Uses **MediaPipe Hands** to detect landmarks in real-time.
- Analyzes finger positions and angles to determine gestures.
- Responds with visual feedback and interaction logic.
- Works inside a Streamlit interface with integrated WebRTC video streaming.

---

## 🙌 Acknowledgements

- MediaPipe
- Streamlit
- OpenCV

---

## ⚠️ Note 
- Make sure your **hand is clearly visible to the camera** at all times for accurate gesture detection. Good lighting helps! 💡  
- Also, ensure your **camera is not mirrored or rotated**, and gestures are performed steadily for optimal recognition. ✋📷
