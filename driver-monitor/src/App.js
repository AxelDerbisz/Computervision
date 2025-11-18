import React, { useRef, useEffect, useState, useCallback } from "react";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";
import { FaceDetection } from "@mediapipe/face_detection";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState({ label: "Loading Model...", confidence: 0 });
  const [isAlerting, setIsAlerting] = useState(false);

  const CONFIDENCE_THRESHOLD = 0.7;
  const SKIP_FRAMES = 10;
  
  const CLASS_NAMES = {
    0: 'DangerousDriving',
    1: 'Distracted',
    2: 'Drinking',
    3: 'SafeDriving',
    4: 'SleepyDriving',
    5: 'Yawn'
  };

  const DANGEROUS_CLASSES = ['DangerousDriving', 'Distracted', 'Drinking', 'SleepyDriving', 'Yawn'];

  const playAlertSound = useCallback(() => {
    if (!isAlerting) return;
    
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();

    oscillator.type = 'sawtooth';
    oscillator.frequency.setValueAtTime(600, audioCtx.currentTime); // Hz
    oscillator.frequency.exponentialRampToValueAtTime(300, audioCtx.currentTime + 0.5);
    
    gainNode.gain.setValueAtTime(0.2, audioCtx.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.5);

    oscillator.connect(gainNode);
    gainNode.connect(audioCtx.destination);

    oscillator.start();
    oscillator.stop(audioCtx.currentTime + 0.5);
  }, [isAlerting]);

  useEffect(() => {
    const loadModel = async () => {
      try {
        const modelUrl = `${process.env.PUBLIC_URL}/model/model.json`;
        
        const loadedModel = await tf.loadGraphModel(modelUrl);
        
        setModel(loadedModel);
        setPrediction({ label: "Model Ready - Start Driving", confidence: 0 });
        console.log("TFJS Graph Model loaded successfully");
      } catch (err) {
        console.error("Failed to load model:", err);
        setPrediction({ label: "Error Loading Model", confidence: 0 });
      }
    };
    loadModel();
  }, []);

  useEffect(() => {
    let frameCount = 0;
    let animationId;
    
    const faceDetection = new FaceDetection({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`,
    });
    
    faceDetection.setOptions({
      model: "short",
      minDetectionConfidence: 0.5,
    });

    faceDetection.onResults((results) => {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      
      ctx.save();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

      if (results.detections.length > 0) {
        const { xCenter, yCenter, width, height } = results.detections[0].boundingBox;
        const x = (xCenter - width / 2) * canvas.width;
        const y = (yCenter - height / 2) * canvas.height;
        const w = width * canvas.width;
        const h = height * canvas.height;

        ctx.strokeStyle = "#00FF00"; 
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);
      } else {
        ctx.fillStyle = "rgba(255, 255, 0, 0.5)";
        ctx.fillRect(0, 0, canvas.width, 50);
        ctx.fillStyle = "black";
        ctx.font = "bold 20px Arial";
        ctx.fillText("⚠️ WARNING: NO FACE DETECTED", 20, 35);
      }
      ctx.restore();
    });

    const runDetection = async () => {
      if (webcamRef.current && webcamRef.current.video.readyState === 4 && model) {
        const video = webcamRef.current.video;

        await faceDetection.send({ image: video });

        if (frameCount % SKIP_FRAMES === 0) {
          tf.tidy(() => {
            const tfImg = tf.browser.fromPixels(video);
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
  }, [model]);

  useEffect(() => {
    if (isAlerting) {
        const interval = setInterval(playAlertSound, 1000); 
        return () => clearInterval(interval);
    }
  }, [isAlerting, playAlertSound]);

  return (
    <div style={styles.container}>
      <h1 style={styles.header}>Driver Behavior Monitor</h1>
      
      <div style={styles.camContainer}>
        <Webcam
          ref={webcamRef}
          style={{ position: "absolute", opacity: 0, width: 640, height: 480 }}
          videoConstraints={{ width: 640, height: 480, facingMode: "user" }}
        />
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
  header: {
    marginBottom: "20px",
  },
  camContainer: {
    position: "relative",
    border: "5px solid #555",
    borderRadius: "10px",
    overflow: "hidden",
    width: "640px",
    height: "480px",
    backgroundColor: "black"
  },
  canvas: {
    width: "100%",
    height: "100%",
    objectFit: "cover",
  },
  alertOverlay: {
    position: "absolute",
    top: "50%",
    left: "50%",
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
    marginTop: "20px",
    padding: "20px",
    backgroundColor: "#4CAF50",
    borderRadius: "10px",
    width: "640px",
    textAlign: "center",
    transition: "background-color 0.3s"
  },
  statusDanger: {
    marginTop: "20px",
    padding: "20px",
    backgroundColor: "#f44336",
    borderRadius: "10px",
    width: "640px",
    textAlign: "center",
    transition: "background-color 0.3s"
  }
};

export default App;