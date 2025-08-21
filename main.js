// This project is based on MediaPipe example code, modified by Alexander Lunt


// Copyright 2023 The MediaPipe Authors.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


// Initialize theme variable
let osColorScheme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";

// Function to update the variable
function updateColorScheme(e) {
  osColorScheme = e.matches ? "dark" : "light";
  console.log("OS color scheme changed:", osColorScheme);
}

// Add listener for changes
const darkModeQuery = window.matchMedia("(prefers-color-scheme: dark)");
darkModeQuery.addEventListener("change", updateColorScheme);


const sequenceLength = 64;
const num_labels = 5;
const numFeatures = 33 * 2; // 33 landmarks * 2 features (x, y)

import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

// const demosSection = document.getElementById("demos");

let poseLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;
const videoHeight = "720px";
const videoWidth = "1280px";

// Before we can use PoseLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
const createPoseLandmarker = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      //modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task`,
      delegate: "GPU"
    },
    runningMode: runningMode,
    numPoses: 1
  });
  // demosSection.classList.remove("invisible");
};
createPoseLandmarker();


// Buffer for last 16 frames of landmarks
let landmarkSequence = [];
let modelSession = null;
let predictionHistory = [0, 0, 0, 0, 0, 0]; // Store last prediction for each state

// Load ONNX model once
async function loadOnnxModel() {
  modelSession = await ort.InferenceSession.create('/models/movement_classifier.onnx');
  console.log('ONNX model loaded:', modelSession);
}
loadOnnxModel();

// Helper: flatten landmarks to [x, y, z, visibility] for all 33 landmarks
function flattenLandmarks(landmarks) {
  // landmarks: array of 33 objects with x, y, z, visibility
  const hip = landmarks[23]; // hip reference
  const hipX = hip.x ?? 0;
  const hipY = hip.y ?? 0;
  // landmarks: array of 33 objects with x, y, z, visibility
  return landmarks.flatMap(lm => [(lm.x ?? 0) - hipX, (lm.y ?? 0) - hipY]);
}


/********************************************************************
// Continuously grab image from webcam stream and detect it.
********************************************************************/

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("outputCanvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);

const cameraSelect = document.getElementById("cameraSelect");
// Populate camera dropdown
function populateCameraDropdown() {
  navigator.mediaDevices.enumerateDevices().then(devices => {
    const cameras = devices.filter(device => device.kind === "videoinput");
    cameraSelect.innerHTML = ""; // Clear previous options
    cameras.forEach((camera, idx) => {
      const option = document.createElement("option");
      option.value = camera.deviceId;
      option.text = camera.label ? String(camera.label) : `Camera ${idx + 1}`;
      cameraSelect.appendChild(option);
    });
  });
}
navigator.mediaDevices.getUserMedia({ video: true }).then(() => {
  populateCameraDropdown();
});

// Check if webcam access is supported.
const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}

// Enable the live webcam view and start detection.
function enableCam(event) {
  if (!poseLandmarker) {
    console.log("Wait! poseLandmarker not loaded yet.");
    return;
  }

  if (webcamRunning === true) {
    webcamRunning = false;
    enableWebcamButton.innerText = "Enable Webcam";

    // Stop the video stream.
    if (video.srcObject) {
      video.srcObject.getTracks().forEach(track => track.stop());
      video.srcObject = null;
    }
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    // checkDevices();
  } else {
    webcamRunning = true;
    enableWebcamButton.innerText = "Disable Webcam";
    // getUsermedia parameters.
    const constraints = {
      video: { deviceId: { exact: cameraSelect.value } }
    };
  
    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
      video.srcObject = stream;
      // video.addEventListener("loadeddata", predictWebcam);
      video.onloadeddata = () => {
        // checkDevices();
        predictWebcam();
      };
    });
  }

}


function checkDevices(){
  // Get all active tracks from the video element's stream
  const tracks = video.srcObject ? video.srcObject.getTracks() : [];
  const activeTracks = tracks.filter(track => track.readyState === "live");

  // Number of active streaming tracks
  console.log("Active media tracks:", activeTracks.length);

  navigator.mediaDevices.enumerateDevices().then(devices => {
  const activeDevices = devices.filter(device => device.kind === "videoinput");
  console.log("Available webcam devices:", activeDevices.length);

  devices.forEach(device => {
    console.log(`Kind: ${device.kind}, Label: ${device.label}, ID: ${device.deviceId}`);
  });


  });
}
function softmax(logits) {
  const maxLogit = Math.max(...logits); // for numerical stability
  const exps = logits.map(x => Math.exp(x - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sumExps);
}

const fullscreenContainer = document.getElementById('fullscreenDiv');


////////////////////////////////////////////////////////////////////////
// Prediction loop
////////////////////////////////////////////////////////////////////////
let lastVideoTime = -1;
async function predictWebcam() {
  if (!webcamRunning || !video.srcObject) {
    return;
  }

  canvasElement.style.height = videoHeight;
  video.style.height = videoHeight;
  canvasElement.style.width = videoWidth;
  video.style.width = videoWidth;
  // Now let's start detecting the stream.
  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await poseLandmarker.setOptions({ runningMode: "VIDEO" });
  }
  let startTimeMs = performance.now();
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    poseLandmarker.detectForVideo(video, startTimeMs, async (result) => {
      canvasCtx.save();
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
      // Flip the image horizontally
      canvasCtx.translate(canvasElement.width, 0);
      canvasCtx.scale(-1, 1);
      // Draw video frame:
      canvasCtx.drawImage(video, 0, 0, canvasElement.width, canvasElement.height);

      // Draw landmarks: 
      for (const landmark of result.landmarks) {
        drawingUtils.drawLandmarks(landmark, {
          radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1)
        });
        drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
      }

      // Collect landmarks for model input
      if (result.landmarks && result.landmarks[0]) {
        // console.log(result.landmarks[0]);  // check the result structure
        const flat = flattenLandmarks(result.landmarks[0]); // flattenLandmarks is a custom function;) it defines which features to use

        // console.log("Flattened landmarks:", flat);
        landmarkSequence.push(flat);
        if (landmarkSequence.length > sequenceLength) landmarkSequence.shift();

        // When we have seqLen frames, run inference
        if (landmarkSequence.length === sequenceLength && modelSession) {
          // Shape: [1, sequenceLength, 66]
          // print landmarkSequence minimum and maximum values
          // console.log("Landmark sequence min:", Math.min(...landmarkSequence.flat()), "Landmark sequence max:", Math.max(...landmarkSequence.flat()));

          
          const inputData = new Float32Array(sequenceLength * numFeatures);
          landmarkSequence.forEach((frame, i) => {
            inputData.set(frame, i * numFeatures);
          });
          
          const inputTensor = new ort.Tensor('float32', inputData, [1, sequenceLength, numFeatures]);
          // console.log("Input tensor shape:", inputTensor.shape);
          // console.log("Input tensor data:", inputTensor.data);
          
          
          // Use correct input name
          const feeds = {};
          feeds[modelSession.inputNames[0]] = inputTensor;
          const output = await modelSession.run(feeds);
          // Assume output is a tensor with shape [1, 6]
          const outputTensor = Object.values(output)[0];
          predictionHistory = Array.from(outputTensor.data);
          
          // Apply softmax
          predictionHistory = softmax(predictionHistory);

          // Update button colors
          stateButtons.forEach((btn, idx) => {
            // Clamp prediction to [0,1] for color
            const red = Math.max(0, Math.min(1, predictionHistory[idx] || 0));
            if (osColorScheme === 'light') {
              btn.style.backgroundColor = `rgb(255, ${Math.round(255 - red * 255)}, ${Math.round(255 - red * 255)})`;
            } else {
              btn.style.backgroundColor = `rgb(${Math.round(red * 255)}, 0, 0)`;
            }
          });
          
          
          // console.log("Model output:", predictionHistory);
        }
      }



      canvasCtx.restore();
    });
    // After drawing (safe! affects only display):
    canvasElement.style.width = "100%"; // match window/container width
    canvasElement.style.height = "auto"; // preserve aspect ratio

  }

  // Call this function again to keep predicting when the browser is ready.
  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam);
  }
}


cameraSelect.addEventListener("change", () => {
  if (webcamRunning) {
    // Stop current stream
    if (video.srcObject) {
      video.srcObject.getTracks().forEach(track => track.stop());
      video.srcObject = null;
    }
    // Start new stream with selected camera
    const constraints = {
      video: { deviceId: { exact: cameraSelect.value } }
    };
    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
      video.srcObject = stream;
      video.onloadeddata = () => {
        checkDevices();
        predictWebcam();
      };
    });
  }
});


// Listening for device changes
navigator.mediaDevices.addEventListener("devicechange", () => {
  navigator.mediaDevices.enumerateDevices().then(devices => {
    const cameras = devices.filter(device => device.kind === "videoinput");
    console.log("Cameras available:", cameras.length);
    cameras.forEach((camera, idx) => {
      console.log(`Camera ${idx + 1}: ${camera.label} (${camera.deviceId})`);
    });
    // Optionally, update your camera dropdown here
    populateCameraDropdown();
  });
});


const stateNames = ['Ruhe', 'Freude', 'Glück', 'Euphorie', 'Ekstase']
const buttonContainer = document.getElementById("stateDiv");
const stateButtons = [];
for (let i = 1; i <= num_labels; i++) {
  const button = document.createElement("button");
  // button.className = "stateButton";
  button.innerText = stateNames[i-1];
  // button.style.color = "black";
  buttonContainer.appendChild(button);
  stateButtons.push(button);
}
