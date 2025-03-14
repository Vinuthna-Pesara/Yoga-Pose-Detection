import React, { useState } from 'react';
import { poseInstructions } from '../../utils/data';
import { poseImages } from '../../utils/pose_images';
import './Instructions.css';

export default function Instructions({ currentPose }) {

    const [instructions] = useState(poseInstructions);

    return (
        <div className="instructions-container">
            <img
                className="pose-demo-img"
                src={poseImages[currentPose]}
                alt={`${currentPose} pose`}  // Adding alt text for better accessibility
            />
            <ul className="instructions-list">
                {instructions[currentPose].map((instruction, index) => (
                    <li key={index} className="instruction">
                        {instruction}
                    </li>
                ))}
            </ul>
        </div>
    );
}
