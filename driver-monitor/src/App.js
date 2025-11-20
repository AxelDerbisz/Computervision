import React, { useRef, useEffect, useState, useCallback } from "react";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";
import { FaceDetection } from "@mediapipe/face_detection";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  
  // --- NOUVEAU : Historique pour le lissage ---
  const predictionHistory = useRef([]); 
  
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState({ label: "Initialisation...", confidence: 0 });
  const [isAlerting, setIsAlerting] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  // --- CONFIGURATION ---
  const CONFIDENCE_THRESHOLD = 0.7;
  const SKIP_FRAMES = 5; // On analyse souvent (toutes les 5 frames)
  const SMOOTHING_BUFFER_SIZE = 10; // On fait la moyenne sur les 10 dernières analyses
  
  const CLASS_NAMES = {
    0: 'Dangerous Driving',
    1: 'Distracted',
    2: 'Drinking',
    3: 'Safe Driving',
    4: 'Sleepy',
    5: 'Yawning'
  };

  const DANGEROUS_CLASSES = ['Dangerous Driving', 'Distracted', 'Drinking', 'Sleepy', 'Yawning'];

  // --- AUDIO (Son Apple Doux) ---
  const playAlertSound = useCallback(() => {
    if (!isAlerting) return;
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();

    oscillator.type = 'sine';
    oscillator.frequency.setValueAtTime(880, audioCtx.currentTime);
    oscillator.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.6);
    
    gainNode.gain.setValueAtTime(0.1, audioCtx.currentTime);
    gainNode.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 0.6);

    oscillator.connect(gainNode);
    gainNode.connect(audioCtx.destination);

    oscillator.start();
    oscillator.stop(audioCtx.currentTime + 0.6);
  }, [isAlerting]);

  // --- LOAD MODEL ---
  useEffect(() => {
    const loadModel = async () => {
      try {
        const modelUrl = `${process.env.PUBLIC_URL}/model/model.json`;
        const loadedModel = await tf.loadGraphModel(modelUrl);
        setModel(loadedModel);
        setPrediction({ label: "Prêt", confidence: 0 });
        setIsLoading(false);
        console.log("System Ready - Smoothing Enabled");
      } catch (err) {
        console.error("Load Error:", err);
        setPrediction({ label: "Erreur Modèle", confidence: 0 });
      }
    };
    loadModel();
  }, []);

  // --- DETECTION LOOP ---
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
      
      // Clean Draw
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

      // Dessin Cadre Visage
      if (results.detections.length > 0) {
        const { xCenter, yCenter, width, height } = results.detections[0].boundingBox;
        const x = (xCenter - width / 2) * canvas.width;
        const y = (yCenter - height / 2) * canvas.height;
        const w = width * canvas.width;
        const h = height * canvas.height;
        const radius = 15;
        ctx.beginPath();
        ctx.lineWidth = 2;
        ctx.strokeStyle = "rgba(255, 255, 255, 0.6)";
        ctx.roundRect(x, y, w, h, radius);
        ctx.stroke();
      }
    });

    const runDetection = async () => {
      if (webcamRef.current && webcamRef.current.video.readyState === 4 && model) {
        const video = webcamRef.current.video;
        await faceDetection.send({ image: video });

        if (frameCount % SKIP_FRAMES === 0) {
          tf.tidy(() => {
            // 1. Préparation Image
            const tfImg = tf.browser.fromPixels(video);
            const resized = tf.image.resizeBilinear(tfImg, [224, 224]);
            const normalized = resized.div(255.0).expandDims(0);
            
            // 2. Prédiction Brute (Instantanée)
            const currentLogits = model.predict(normalized).dataSync(); // Float32Array des probas

            // --- ALGORITHME DE LISSAGE (SMOOTHING) ---
            
            // A. Ajouter au buffer
            predictionHistory.current.push(currentLogits);
            
            // B. Garder seulement les N dernières prédictions
            if (predictionHistory.current.length > SMOOTHING_BUFFER_SIZE) {
                predictionHistory.current.shift();
            }

            // C. Calculer la moyenne de chaque classe sur l'historique
            const numClasses = Object.keys(CLASS_NAMES).length;
            const averagedLogits = new Float32Array(numClasses).fill(0);

            for (let entry of predictionHistory.current) {
                for (let i = 0; i < numClasses; i++) {
                    averagedLogits[i] += entry[i];
                }
            }
            
            // Diviser par la taille de l'historique pour avoir la moyenne
            for (let i = 0; i < numClasses; i++) {
                averagedLogits[i] /= predictionHistory.current.length;
            }

            // 3. Décision finale basée sur la MOYENNE
            const maxIndex = averagedLogits.indexOf(Math.max(...averagedLogits));
            const label = CLASS_NAMES[maxIndex];
            const confidence = averagedLogits[maxIndex];

            setPrediction({ label, confidence });

            // 4. Déclenchement Alerte
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

  // --- SOUND TRIGGER ---
  useEffect(() => {
    if (isAlerting) {
        const interval = setInterval(playAlertSound, 1000); 
        return () => clearInterval(interval);
    }
  }, [isAlerting, playAlertSound]);

  // --- RENDER (Style Apple) ---
  return (
    <div style={styles.pageContainer}>
      <style>
        {`
          @keyframes pulse-red {
            0% { box-shadow: 0 0 0 0 rgba(255, 59, 48, 0.4); }
            70% { box-shadow: 0 0 0 20px rgba(255, 59, 48, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 59, 48, 0); }
          }
          @keyframes fade-in {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
          }
        `}
      </style>

      <div style={styles.mainCard}>
        <div style={styles.header}>
          <div style={styles.headerIcon}>🚗</div>
          <h1 style={styles.title}>Driver<span style={{fontWeight: 400, opacity: 0.7}}>Guard</span></h1>
        </div>

        <div style={{
            ...styles.videoWrapper,
            ...(isAlerting ? styles.videoWrapperAlert : {})
        }}>
          <Webcam
            ref={webcamRef}
            style={styles.webcam}
            videoConstraints={{ width: 640, height: 480, facingMode: "user" }}
            muted
            playsInline
          />
          <canvas ref={canvasRef} width={640} height={480} style={styles.canvas} />
          
          {isAlerting && (
             <div style={styles.alertGlass}>
                <span style={{fontSize: '3rem'}}>⚠️</span>
                <span style={styles.alertText}>{prediction.label.toUpperCase()}</span>
             </div>
          )}

          {isLoading && (
              <div style={styles.loadingOverlay}>Chargement de l'IA...</div>
          )}
        </div>

        <div style={styles.dashboard}>
          <div style={styles.statusRow}>
            <div style={styles.labelGroup}>
              <span style={styles.subLabel}>ÉTAT LISSÉ</span>
              <span style={{
                  ...styles.mainLabel,
                  color: isAlerting ? '#FF3B30' : '#34C759'
              }}>
                {prediction.label}
              </span>
            </div>
            
            <div style={styles.confidenceGroup}>
              <span style={styles.subLabel}>STABILITÉ</span>
              <div style={styles.progressBarBg}>
                <div style={{
                    ...styles.progressBarFill,
                    width: `${(prediction.confidence * 100).toFixed(0)}%`,
                    backgroundColor: isAlerting ? '#FF3B30' : '#34C759'
                }} />
              </div>
              <span style={styles.confidenceValue}>{(prediction.confidence * 100).toFixed(0)}%</span>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}

// --- STYLES ---
const styles = {
  pageContainer: {
    minHeight: "100vh",
    backgroundColor: "#000000",
    backgroundImage: "radial-gradient(circle at 50% 0%, #2c2c2e 0%, #000000 100%)",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif",
    color: "#FFFFFF",
    padding: "20px",
    boxSizing: "border-box"
  },
  mainCard: {
    width: "100%",
    maxWidth: "680px",
    display: "flex",
    flexDirection: "column",
    gap: "24px",
    animation: "fade-in 0.8s ease-out"
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "12px",
    marginBottom: "10px"
  },
  headerIcon: {
    fontSize: "24px",
    background: "rgba(255,255,255,0.1)",
    borderRadius: "50%",
    width: "40px",
    height: "40px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    backdropFilter: "blur(10px)"
  },
  title: {
    margin: 0,
    fontSize: "28px",
    fontWeight: "700",
    letterSpacing: "-0.5px",
  },
  videoWrapper: {
    position: "relative",
    borderRadius: "28px",
    overflow: "hidden",
    boxShadow: "0 20px 40px rgba(0,0,0,0.4)",
    border: "1px solid rgba(255,255,255,0.1)",
    backgroundColor: "#1c1c1e",
    aspectRatio: "4/3",
    transition: "all 0.3s ease"
  },
  videoWrapperAlert: {
    border: "2px solid #FF3B30",
    animation: "pulse-red 1.5s infinite"
  },
  webcam: {
    width: "100%",
    height: "100%",
    objectFit: "cover",
    display: "block"
  },
  canvas: {
    position: "absolute", top: 0, left: 0, width: "100%", height: "100%", objectFit: "cover"
  },
  alertGlass: {
    position: "absolute", top: 0, left: 0, right: 0, bottom: 0,
    display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
    background: "rgba(50, 0, 0, 0.4)", backdropFilter: "blur(8px)", zIndex: 10, gap: "10px"
  },
  alertText: {
    fontSize: "24px", fontWeight: "700", color: "#FFFFFF", textShadow: "0 2px 10px rgba(0,0,0,0.5)", letterSpacing: "1px"
  },
  loadingOverlay: {
    position: "absolute", top: 0, left: 0, right: 0, bottom: 0,
    display: "flex", alignItems: "center", justifyContent: "center", background: "#1c1c1e", zIndex: 20, color: "#8e8e93"
  },
  dashboard: {
    background: "rgba(28, 28, 30, 0.6)", backdropFilter: "blur(20px)", WebkitBackdropFilter: "blur(20px)",
    borderRadius: "24px", padding: "20px 24px", border: "1px solid rgba(255, 255, 255, 0.08)", boxShadow: "0 8px 32px rgba(0, 0, 0, 0.2)"
  },
  statusRow: {
    display: "flex", justifyContent: "space-between", alignItems: "center"
  },
  labelGroup: {
    display: "flex", flexDirection: "column", gap: "4px"
  },
  subLabel: {
    fontSize: "11px", textTransform: "uppercase", color: "#8e8e93", fontWeight: "600", letterSpacing: "0.5px"
  },
  mainLabel: {
    fontSize: "20px", fontWeight: "600", letterSpacing: "-0.5px", transition: "color 0.3s ease"
  },
  confidenceGroup: {
    display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "6px", width: "40%"
  },
  progressBarBg: {
    width: "100%", height: "6px", background: "rgba(255,255,255,0.1)", borderRadius: "10px", overflow: "hidden"
  },
  progressBarFill: {
    height: "100%", borderRadius: "10px", transition: "width 0.3s ease, background-color 0.3s ease"
  },
  confidenceValue: {
    fontSize: "12px", color: "#8e8e93", fontVariantNumeric: "tabular-nums"
  }
};

export default App;