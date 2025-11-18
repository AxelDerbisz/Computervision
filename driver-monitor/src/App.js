import React, { useRef, useEffect, useState, useCallback } from "react";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";
import { FaceDetection } from "@mediapipe/face_detection";

function App() {
  const webcamRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState({ label: "Loading Model...", confidence: 0 });
  const [isAlerting, setIsAlerting] = useState(false);
  const [videoSource, setVideoSource] = useState(null); // State for uploaded video

  // --- CONFIGURATION ---
  const CONFIDENCE_THRESHOLD = 0.7; 
  const SKIP_FRAMES = 5; // Process every 5th frame
  
  const CLASS_NAMES = {
    0: 'DangerousDriving',
    1: 'Distracted',
    2: 'Drinking',
    3: 'SafeDriving',
    4: 'SleepyDriving',
    5: 'Yawn'
  };

  const DANGEROUS_CLASSES = ['DangerousDriving', 'Distracted', 'Drinking', 'SleepyDriving', 'Yawn'];

  // --- SOUND ALARM ---
  const playAlertSound = useCallback(() => {
    if (!isAlerting) return;
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();
    oscillator.type = 'sawtooth';
    oscillator.frequency.setValueAtTime(600, audioCtx.currentTime); 
    oscillator.frequency.exponentialRampToValueAtTime(300, audioCtx.currentTime + 0.5);
    gainNode.gain.setValueAtTime(0.2, audioCtx.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.5);
    oscillator.connect(gainNode);
    gainNode.connect(audioCtx.destination);
    oscillator.start();
    oscillator.stop(audioCtx.currentTime + 0.5);
  }, [isAlerting]);

  // --- 1. LOAD MODEL ---
  useEffect(() => {
    const loadModel = async () => {
      try {
        const modelUrl = `${process.env.PUBLIC_URL}/model/model.json`;
        // Using loadGraphModel as fixed previously
        const loadedModel = await tf.loadGraphModel(modelUrl);
        setModel(loadedModel);
        setPrediction({ label: "Model Ready - Select Source", confidence: 0 });
        console.log("TFJS Graph Model loaded successfully");
      } catch (err) {
        console.error("Failed to load model:", err);
        setPrediction({ label: "Error Loading Model", confidence: 0 });
      }
    };
    loadModel();
  }, []);

  // --- 2. HANDLE VIDEO UPLOAD ---
  const handleVideoUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setVideoSource(url);
      setPrediction({ label: "Video Loaded - Playing", confidence: 0 });
    }
  };

  // --- 3. DETECTION LOOP ---
  useEffect(() => {
    let frameCount = 0;
    let animationId;
    
    const faceDetection = new FaceDetection({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`,
    });
    
    faceDetection.setOptions({ model: "short", minDetectionConfidence: 0.5 });

    faceDetection.onResults((results) => {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      
      // Draw Video Frame on Canvas
      ctx.save();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

      // Draw Face Box
      if (results.detections.length > 0) {
        const { xCenter, yCenter, width, height } = results.detections[0].boundingBox;
        const x = (xCenter - width / 2) * canvas.width;
        const y = (yCenter - height / 2) * canvas.height;
        const w = width * canvas.width;
        const h = height * canvas.height;
        ctx.strokeStyle = "#00FF00"; 
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);
      }
      ctx.restore();
    });

    const runDetection = async () => {
      // DETERMINE SOURCE: Video File OR Webcam
      let inputElement = null;

      if (videoSource && videoRef.current && !videoRef.current.paused) {
         inputElement = videoRef.current;
      } else if (!videoSource && webcamRef.current && webcamRef.current.video?.readyState === 4) {
         inputElement = webcamRef.current.video;
      }

      if (inputElement && model) {
        // A. Run MediaPipe
        await faceDetection.send({ image: inputElement });

        // B. Run Driver Behavior Model
        if (frameCount % SKIP_FRAMES === 0) {
          tf.tidy(() => {
            const tfImg = tf.browser.fromPixels(inputElement);
            const resized = tf.image.resizeBilinear(tfImg, [224, 224]);
            const normalized = resized.div(255.0).expandDims(0);
            
            const predictions = model.predict(normalized).dataSync();
            const maxIndex = predictions.indexOf(Math.max(...predictions));
            const label = CLASS_NAMES[maxIndex];
            const confidence = predictions[maxIndex];

            setPrediction({ label, confidence });

            if (DANGEROUS_CLASSES.includes(label) && confidence > CONFIDENCE_THRESHOLD) {
               setIsAlerting(true);
            } else {
               setIsAlerting(false);
            }
          });
        }
        frameCount++;
      }
      animationId = requestAnimationFrame(runDetection);
    };

    runDetection();
    return () => cancelAnimationFrame(animationId);
  }, [model, videoSource]); // Re-run if video source changes

  useEffect(() => {
    if (isAlerting) {
        const interval = setInterval(playAlertSound, 1000);
        return () => clearInterval(interval);
    }
  }, [isAlerting, playAlertSound]);

  return (
    <div style={styles.container}>
      <h1 style={styles.header}>Driver Behavior Monitor</h1>
      
      {/* Controls */}
      <div style={{ marginBottom: "10px" }}>
        <input type="file" accept="video/*" onChange={handleVideoUpload} />
        {videoSource && <button onClick={() => setVideoSource(null)}>Switch back to Webcam</button>}
      </div>

      <div style={styles.camContainer}>
        {/* 1. Webcam (Hidden if video is selected) */}
        {!videoSource && (
            <Webcam
            ref={webcamRef}
            style={{ position: "absolute", opacity: 0, width: 640, height: 480 }}
            videoConstraints={{ width: 640, height: 480, facingMode: "user" }}
            />
        )}

        {/* 2. Video Player (Visible if file selected) */}
        {videoSource && (
            <video 
                ref={videoRef}
                src={videoSource}
                controls 
                loop
                playsInline
                width="640"
                height="480"
                style={{ position: "absolute", opacity: 0 }} // Hide raw video, show canvas
                onPlay={() => console.log("Video playing...")}
            />
        )}
        
        {/* 3. Canvas (Always visible output) */}
        <canvas ref={canvasRef} width={640} height={480} style={styles.canvas} />
        
        {isAlerting && (
            <div style={styles.alertOverlay}>
                🚨 {prediction.label.toUpperCase()} 🚨
            </div>
        )}
      </div>

      <div style={isAlerting ? styles.statusDanger : styles.statusSafe}>
        <h2>Status: {prediction.label}</h2>
        <p>Confidence: {(prediction.confidence * 100).toFixed(1)}%</p>
      </div>
    </div>
  );
}

const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    backgroundColor: "#282c34",
    minHeight: "100vh",
    color: "white",
    fontFamily: "Arial, sans-serif",
    padding: "20px"
  },
  header: { marginBottom: "20px" },
  camContainer: {
    position: "relative",
    border: "5px solid #555",
    borderRadius: "10px",
    overflow: "hidden",
    width: "640px",
    height: "480px",
    backgroundColor: "black"
  },
  canvas: { width: "100%", height: "100%", objectFit: "cover" },
  alertOverlay: {
    position: "absolute",
    top: "50%", left: "50%",
    transform: "translate(-50%, -50%)",
    backgroundColor: "rgba(255, 0, 0, 0.7)",
    color: "white",
    padding: "20px",
    fontSize: "2rem",
    fontWeight: "bold",
    borderRadius: "10px",
    animation: "blink 1s infinite"
  },
  statusSafe: {
    marginTop: "20px", padding: "20px",
    backgroundColor: "#4CAF50", borderRadius: "10px",
    width: "640px", textAlign: "center", transition: "background-color 0.3s"
  },
  statusDanger: {
    marginTop: "20px", padding: "20px",
    backgroundColor: "#f44336", borderRadius: "10px",
    width: "640px", textAlign: "center", transition: "background-color 0.3s"
  }
};

export default App;