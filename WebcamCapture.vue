<template>
  <div class="video-container">
    <h2>Webcam Feed</h2>
    <video ref="videoElement" autoplay muted playsinline></video>
    <canvas ref="canvasElement" class="overlay-canvas"></canvas>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import * as tf from '@tensorflow/tfjs';
import * as blazeface from '@tensorflow-models/blazeface';

// Reactive references for our DOM elements
const videoElement = ref<HTMLVideoElement | null>(null);
const canvasElement = ref<HTMLCanvasElement | null>(null);

// Variables for the model and animation frame
let model: blazeface.BlazeFaceModel | null = null;
let animationFrameId: number;

// Liveness detection variables
let previousFacePosition: [number, number] | null = null;
let isLive = false;
const movementThreshold = 15; // Pixels of movement required to be considered 'live'
let stillnessFrames = 0;
const stillnessThreshold = 30; // Number of frames to remain still before turning red
let blinkDetected = false;
let previousEyeDistance = 0;
const blinkThreshold = 0.5; // A factor to detect a significant change in eye distance

// Function to load the face detection model
const loadModel = async () => {
  console.log("Loading blazeface model...");
  model = await blazeface.load();
  console.log("Blazeface model loaded.");
};

// Function to draw a bounding box on the canvas
const drawDetections = (detections: blazeface.NormalizedFace[]) => {
  if (!canvasElement.value || !videoElement.value) return;

  const ctx = canvasElement.value.getContext('2d');
  if (!ctx) return;

  ctx.clearRect(0, 0, canvasElement.value.width, canvasElement.value.height);
  canvasElement.value.width = videoElement.value.videoWidth;
  canvasElement.value.height = videoElement.value.videoHeight;
  
  // A small factor to scale the box and a vertical offset to adjust its position
  const scaleFactor = 1.2;
  const yOffset = 20;

  detections.forEach(detection => {
    const [xStart, yStart] = detection.topLeft as [number, number];
    const [xEnd, yEnd] = detection.bottomRight as [number, number];

    const originalWidth = xEnd - xStart;
    const originalHeight = yEnd - yStart;

    const newWidth = originalWidth * scaleFactor;
    const newHeight = originalHeight * scaleFactor;

    const newXStart = xStart - (newWidth - originalWidth) / 2;
    const newYStart = yStart - (newHeight - originalHeight) / 2 + yOffset;

    ctx.beginPath();
    ctx.rect(newXStart, newYStart, newWidth, newHeight);
    ctx.lineWidth = 2;
    ctx.strokeStyle = isLive ? '#00ff00' : '#ff0000';
    ctx.stroke();
  });
};

// Main function to detect faces and run the animation loop
const detectFaces = async () => {
  if (!model || !videoElement.value || videoElement.value.readyState < 2) {
    animationFrameId = requestAnimationFrame(detectFaces);
    return;
  }
  const faces = await model.estimateFaces(videoElement.value, false);
  
  if (faces.length > 0) {
    const face = faces[0];
    const [xStart, yStart] = face.topLeft as [number, number];
    const facePosition = [xStart, yStart];

    // 1. Movement Check
    if (previousFacePosition) {
      const dx = Math.abs(facePosition[0] - previousFacePosition[0]);
      const dy = Math.abs(facePosition[1] - previousFacePosition[1]);
      const movement = Math.sqrt(dx * dx + dy * dy);

      if (movement > movementThreshold) {
        isLive = true;
        stillnessFrames = 0;
      } else {
        stillnessFrames++;
        if (stillnessFrames > stillnessThreshold) {
          isLive = false;
        }
      }
    }
    previousFacePosition = facePosition as [number, number];
    
    // 2. Blink Check (with a check for landmarks)
    if (face.landmarks) {
      const rightEye = face.landmarks[0] as [number, number];
      const leftEye = face.landmarks[2] as [number, number];
      const eyeDistance = Math.abs(rightEye[1] - leftEye[1]);

      if (previousEyeDistance > 0) {
        // Log the values for debugging
        console.log(`Eye Distances: current=${eyeDistance.toFixed(2)}, previous=${previousEyeDistance.toFixed(2)}`);
        
        if (eyeDistance < previousEyeDistance * blinkThreshold) {
            console.log("BLINK DETECTED!");
            isLive = true;
            stillnessFrames = 0;
        }
      }
      previousEyeDistance = eyeDistance;
    }
    
  } else {
    isLive = false;
    previousFacePosition = null;
    stillnessFrames = 0;
    previousEyeDistance = 0;
  }

  drawDetections(faces);
  animationFrameId = requestAnimationFrame(detectFaces);
};

// This function will run after the component is mounted to the DOM
onMounted(async () => {
  try {
    if (videoElement.value) {
      console.log("Initializing TensorFlow.js backend...");
      await tf.setBackend('webgl');
      await tf.ready();
      console.log("TensorFlow.js backend initialized.");
      
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoElement.value.srcObject = stream;
      
      videoElement.value.onloadedmetadata = async () => {
        await loadModel();
        detectFaces();
      };
    }
  } catch (err) {
    console.error("Error accessing the webcam or loading the model:", err);
    alert("Could not access the webcam or load the face detection model.");
  }
});
</script>

<style scoped>
.video-container {
  position: relative;
  width: fit-content;
  margin: auto;
}

video {
  display: block;
  width: 100%;
  max-width: 640px;
  border: 2px solid #ccc;
  border-radius: 8px;
}

.overlay-canvas {
  position: absolute;
  top: 0;
  left: 0;
}
</style>

