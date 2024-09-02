import './App.css';
import React, { useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import Webcam from 'react-webcam';
import '@tensorflow/tfjs-backend-webgl';
import * as poseDetection from '@tensorflow-models/pose-detection';


function App() {
  // basic Reference
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  // initializing variables
  var pos;
  var stage;
  var counter = 0;

  var r_pos;
  var r_stage;
  var r_counter = 0;

  var final_counter = 0;

  // basic functions
  const slope = (x1, y1, x2, y2) => {
    return (y2 - y1) / (x2 - x1);
  }
  const distance = (x1, y1, x2, y2) => {
    return Math.pow(Math.pow(y2 - y1, 2) + Math.pow(x2 - x1, 2), 0.5)
  }
  const roundoff = (x1, y1, x2, y2) => {
    const s = slope(x1, y1, x2, y2);
    const d = 1
    const x = Math.pow(d * ((Math.pow(d, 2)) / (1 + Math.pow(s, 2))), 0.5)
    const y = s * x
    //return -round(x, 3), -round(y, 3)
    return [-x.toFixed(3), -y.toFixed(3)]
  }
  const retrivew = (x1, y1, x2, y2, x3, y3, x4, y4) => {
    const s = slope(x1, y1, x2, y2)
    const d = distance(x3, y3, x4, y4)
    const x = Math.pow(d * (1 / (1 + Math.pow(s, 2))), 0.5) + x3
    const y = s * (x - x3) + y3
    return [x, y]
  }
  const angle = (x1, y1, x2, y2, x3, y3) => {
    var dAx = x2 - x1;
    var dAy = y2 - y1;
    var dBx = x3 - x2;
    var dBy = y3 - y2;
    var ans = Math.atan2(dAx * dBy - dAy * dBx, dAx * dBx + dAy * dBy);
    if (ans < 0) { ans = ans * -1; }
    var final_ans = ans * (180 / Math.PI);
    return final_ans;
  }

  // load and run handpose models
  const runHandpose = async () => {
    console.log("loading models")
    await tf.ready();
    const detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet);
    console.log('Movenet model loaded.');
    const model = await tf.loadLayersModel('https://raw.githubusercontent.com/sufiyanpatel27/burnwise/models/d_pull_ups/classifier/model.json')
    console.log('classification model loaded')
    const model2 = await tf.loadLayersModel('https://raw.githubusercontent.com/sufiyanpatel27/burnwise/models/d_pull_ups/regressor/model.json')
    console.log('regression model loaded')
    try {
      setInterval(() => {
        detect(detector, model, model2)
      }, 1)
    } catch (error) {
      console.log('nothing')
    }
  };

  runHandpose();

  // defining main logic function - detect
  const detect = async (m_model, c_model, r_model) => {
    if (
      typeof webcamRef.current !== 'undefined' &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      const hand = await m_model.estimatePoses(video);

      const ctx = canvasRef.current.getContext('2d');

      if (hand.length > 0) {

        var coords = [
          [hand[0].keypoints[5].x, hand[0].keypoints[5].y],
          [hand[0].keypoints[6].x, hand[0].keypoints[6].y],
          [hand[0].keypoints[7].x, hand[0].keypoints[7].y],
          [hand[0].keypoints[8].x, hand[0].keypoints[8].y],
          [hand[0].keypoints[9].x, hand[0].keypoints[9].y],
          [hand[0].keypoints[10].x, hand[0].keypoints[10].y],
        ]

        ctx.moveTo(coords[0][0], coords[0][1]);
        ctx.lineTo(coords[1][0], coords[1][1]);

        ctx.moveTo(coords[0][0], coords[0][1]);
        ctx.lineTo(coords[2][0], coords[2][1]);

        ctx.moveTo(coords[1][0], coords[1][1]);
        ctx.lineTo(coords[3][0], coords[3][1]);

        ctx.moveTo(coords[2][0], coords[2][1]);
        ctx.lineTo(coords[4][0], coords[4][1]);

        ctx.moveTo(coords[3][0], coords[3][1]);
        ctx.lineTo(coords[5][0], coords[5][1]);

        ctx.stroke();


        for (var i = 0; i < coords.length; i++) {
          ctx.beginPath();
          ctx.arc(coords[i][0], coords[i][1], 5, 0, Math.PI * 2, true);
          ctx.fillStyle = "#00FF33"
          ctx.fill();
        }

        //left hand
        const x = roundoff(hand[0].keypoints[5].x, hand[0].keypoints[5].y, hand[0].keypoints[7].x, hand[0].keypoints[7].y)
        let input_xs = tf.tensor2d([[x[0], x[1]]]);
        let output = c_model.predict(input_xs);
        const outputData = output.dataSync();
        //right hand
        const right_x = roundoff(hand[0].keypoints[6].x, hand[0].keypoints[6].y, hand[0].keypoints[8].x, hand[0].keypoints[8].y)
        let right_input_xs = tf.tensor2d([[right_x[0], right_x[1]]]);
        let right_output = c_model.predict(right_input_xs);
        const right_outputData = right_output.dataSync();


        //left wrist prediction
        const ans = r_model.predict(input_xs)
        const outputData_1 = ans.dataSync();
        const x_reg = retrivew(x[0], x[1], outputData_1[0], outputData_1[1], hand[0].keypoints[7].x, hand[0].keypoints[7].y, hand[0].keypoints[9].x, hand[0].keypoints[9].y)

        //right wrist prediction
        const right_ans = r_model.predict(right_input_xs)
        const right_outputData_1 = right_ans.dataSync();
        const right_x_reg = retrivew(right_x[0], right_x[1], right_outputData_1[0], right_outputData_1[1], hand[0].keypoints[8].x, hand[0].keypoints[8].y, hand[0].keypoints[10].x, hand[0].keypoints[9].y)


        if (outputData[0] == 1) {
          if ((x_reg[0] - hand[0].keypoints[9].x).toFixed(2) > 50) {
            document.getElementById('left_arm_sugg').innerHTML = "move left"
            document.getElementById('left_arm_sugg').style.color = 'red'
            ctx.beginPath();
            ctx.arc(coords[4][0], coords[4][1], 10, 0, Math.PI * 2, true);
            ctx.fillStyle = "red"
            ctx.fill();
          }
          else if ((x_reg[0] - hand[0].keypoints[9].x).toFixed(2) < -50) {
            document.getElementById('left_arm_sugg').innerHTML = "move right"
            document.getElementById('left_arm_sugg').style.color = 'red'
            ctx.beginPath();
            ctx.arc(coords[4][0], coords[4][1], 10, 0, Math.PI * 2, true);
            ctx.fillStyle = "red"
            ctx.fill();
          }
          else {
            document.getElementById('left_arm_sugg').innerHTML = "good"
            document.getElementById('left_arm_sugg').style.color = '#00FF33'
            const ans = angle(coords[0][0], coords[0][1], coords[2][0], coords[2][1], coords[4][0], coords[4][1]);
            if (ans > 90) {
              const x = roundoff(hand[0].keypoints[5].x, hand[0].keypoints[5].y, hand[0].keypoints[7].x, hand[0].keypoints[7].y + 20)
              let input_xs = tf.tensor2d([[x[0], x[1]]]);
              let output = c_model.predict(input_xs);
              const outputData = output.dataSync();
              //console.log(outputData)
              pos = "down"
            }
            else {
              const x = roundoff(hand[0].keypoints[5].x, hand[0].keypoints[5].y, hand[0].keypoints[7].x, hand[0].keypoints[7].y - 20)
              let input_xs = tf.tensor2d([[x[0], x[1]]]);
              let output = c_model.predict(input_xs);
              const outputData = output.dataSync();
              //console.log(outputData)
              pos = "up"
            }
            if (pos == "down") {
              stage = "down"
            }
            if (pos == "up" && stage == "down") {
              stage = "up"
              counter += 1
              document.getElementById('final_counter').innerHTML = counter;
            }
          }
        }
        else {
          document.getElementById('left_arm_sugg').innerHTML = "elbow error"
          document.getElementById('left_arm_sugg').style.color = 'red'
          ctx.beginPath();
          ctx.arc(coords[2][0], coords[2][1], 10, 0, Math.PI * 2, true);
          ctx.fillStyle = "red"
          ctx.fill();
        }

        if (right_outputData[0] == 1) {
          if ((right_x_reg[0] - hand[0].keypoints[10].x).toFixed(2) > 50) {
            document.getElementById('right_arm_sugg').innerHTML = "move left"
            document.getElementById('right_arm_sugg').style.color = 'red'
            ctx.beginPath();
            ctx.arc(coords[5][0], coords[5][1], 10, 0, Math.PI * 2, true);
            ctx.fillStyle = "red"
            ctx.fill();
          }
          else if ((right_x_reg[0] - hand[0].keypoints[10].x).toFixed(2) < -50) {
            document.getElementById('right_arm_sugg').innerHTML = "move right"
            document.getElementById('right_arm_sugg').style.color = 'red'
            ctx.beginPath();
            ctx.arc(coords[5][0], coords[5][1], 10, 0, Math.PI * 2, true);
            ctx.fillStyle = "red"
            ctx.fill();
          }
          else {
            document.getElementById('right_arm_sugg').innerHTML = "good"
            document.getElementById('right_arm_sugg').style.color = '#00FF33'
            const ans = angle(coords[1][0], coords[1][1], coords[3][0], coords[3][1], coords[5][0], coords[5][1]);
            if (ans > 90) {
              const x = roundoff(hand[0].keypoints[6].x, hand[0].keypoints[6].y, hand[0].keypoints[8].x, hand[0].keypoints[8].y + 20)
              let input_xs = tf.tensor2d([[x[0], x[1]]]);
              let output = c_model.predict(input_xs);
              const outputData = output.dataSync();
              //console.log(outputData)
              r_pos = "down"
            }
            else {
              const x = roundoff(hand[0].keypoints[6].x, hand[0].keypoints[6].y, hand[0].keypoints[8].x, hand[0].keypoints[8].y - 20)
              let input_xs = tf.tensor2d([[x[0], x[1]]]);
              let output = c_model.predict(input_xs);
              const outputData = output.dataSync();
              //console.log(outputData)
              r_pos = "up"
            }
            if (r_pos == "down") {
              r_stage = "down"
            }
            if (r_pos == "up" && r_stage == "down") {
              r_stage = "up"
              r_counter += 1
              //document.getElementById('final_counter').innerHTML = counter;
              if (counter == r_counter) {
                final_counter += 1;
                document.getElementById('final_counter').innerHTML = final_counter;
              }
              /*else {
                alert("here")
              }*/
            }
          }
        }
        else {
          document.getElementById('right_arm_sugg').innerHTML = "elbow error"
          document.getElementById('right_arm_sugg').style.color = 'red'
          ctx.beginPath();
          ctx.arc(coords[3][0], coords[3][1], 10, 0, Math.PI * 2, true);
          ctx.fillStyle = "red"
          ctx.fill();
        }
      }
    }
  }








  return (
    <div className="App">
      <div className="left-panel">
        <Webcam
          ref={webcamRef}
          style={{
            position: 'absolute',
            left: 0,
            right: 0,
            textAlign: 'center',
            zindex: 9,
            width: "70%",
            height: "100%"
          }} />
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute',
            left: 0,
            right: 0,
            textAlign: 'center',
            zindex: 9,
            width: "70%",
            height: "100%"
          }} />
      </div>
      <div className="right-panel">
        <div className="exercise_name">
          <h1 style={{ color: "#00FF33" }}>Dumbbell Shoulder Press</h1>
        </div>
        <div className="active_data_points">
          <div className="head">
            <h2 style={{ color: "white" }}>Active Data Points</h2>
          </div>
          <div className="points" style={{ color: "#00FF33" }}>
            <div className="left_points">
              <h5>left wrist</h5>
              <h5>left elbow</h5>
              <h5>left shoulder</h5>
            </div>
            <div className="right_points">
              <h5>right wrist</h5>
              <h5>right elbow</h5>
              <h5>right shoulder</h5>
            </div>
          </div>
        </div>
        <div className="counter">
          <div className="counter_head">
            <h2 style={{ color: "white" }}>Counter</h2>
          </div>
          <div className="counter_count">
            <h3 id="final_counter" style={{ color: "#00FF33" }}>loading models</h3>
          </div>
        </div>
        <div className="suggesstions">
          <div className="left_arm">
            <h4>Left Arm : </h4>
            <h2 style={{ color: "#00FF33" }} id="left_arm_sugg">loading models</h2>
          </div>
          <div className="left_arm">
            <h4>Right Arm : </h4>
            <h2 style={{ color: "#00FF33" }} id="right_arm_sugg">loading models</h2>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
