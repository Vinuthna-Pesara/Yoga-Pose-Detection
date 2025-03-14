import React from 'react'
import { Link } from 'react-router-dom'

import './Home.css'

export default function Home() {

    return (
        <div className='home-container'>
            <div className='home-header'>
                <h1 className='home-heading'>YogaTrainer</h1>
                <div className="btn-container">
                <Link to='/about'>
                    <button 
                        className="btn btn-secondary" 
                        id="header-btn"
                    >
                        About
                    </button>
                </Link>
                <Link to='/tutorials'>
                        <button
                            className="btn btn-secondary"
                            id="header-btn"
                        >Tutorials</button>
                    </Link>
            </div>
            </div>
            <div className="home-main">
                <div className="btn-section">
                    <Link to='/start'>
                        <button
                            className="btn start-btn"
                        >Let's Start</button>
                    </Link>

                    <Link to='/PoseDetection'>
                        <button
                            className="btn start-btn"
                        >PoseDetection</button>
                    </Link>

                </div>
            </div>
        </div>
    )
}
