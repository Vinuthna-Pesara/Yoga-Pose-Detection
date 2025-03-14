import React from 'react'
import './About.css'

export default function About() {
    return (
        <div className="about-container">
        <h1 className="about-heading">About</h1>
            <div className="about-main">
                <p className="about-content">
                    This is a real-time AI-based Yoga Trainer that detects how well you're performing your yoga poses.
                    I created this project as a personal initiative, and it’s deployed for public use.
                    It’s mainly aimed at helping developers learning AI to explore and improve their skills.
                    The project is open source, and the code is available on GitHub:
                    <a href="https://github.com/harshbhatt7585/YogaIntelliJ" target="_blank" rel="noopener noreferrer">
                        https://github.com/harshbhatt7585/YogaIntelliJ
                    </a>
                </p>
                <p className="about-content">
                    The AI first predicts keypoints or coordinates of different body parts (i.e., where they appear in an image).
                    Then, it uses a classification model to identify whether you're performing the correct pose.
                    If the AI detects the pose with more than 95% accuracy, it will notify you by turning the virtual skeleton green,
                    signaling that the pose is correct.
                </p>
                <p className="about-content">
                    I’ve used TensorFlow’s pretrained MoveNet model to predict the keypoints and built a neural network on top of it
                    to classify the yoga poses using these coordinates.
                    The model was initially trained in Python, but thanks to TensorFlow.js, I converted the model for browser support
                    so it can run directly in the web environment.
                </p>
            </div>
        </div>
    )
}
