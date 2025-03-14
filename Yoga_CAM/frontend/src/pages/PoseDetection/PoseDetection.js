import React, { useState } from 'react';
import axios from 'axios';
import './PoseDetection.css'; // Import the CSS file

export default function PoseDetection() {
  const [image, setImage] = useState(null);
  const [pose, setPose] = useState('');
  const [imagePreview, setImagePreview] = useState('');
  const [annotatedImagePath, setAnnotatedImagePath] = useState('');
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setImagePreview(URL.createObjectURL(file));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) {
      alert('Please select an image');
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append('image', image);

    try {
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setPose(response.data.pose);
      setAnnotatedImagePath(response.data.imagePath);
    } catch (error) {
      console.error('Error uploading image:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
      <div className="pose-detection-container">
        <h2>Pose Detection</h2>
        <form className="pose-detection-form" onSubmit={handleSubmit}>
          <input type="file" accept="image/*" onChange={handleImageChange} />
          <button type="submit" disabled={loading}>
            {loading ? 'Processing...' : 'Detect Pose'}
          </button>
        </form>

        <div className="preview-container">
          {imagePreview && (
            <div>
              <h3>Uploaded Image:</h3>
              <img src={imagePreview} alt="Uploaded preview" />
            </div>
          )}

          {annotatedImagePath && (
            <div>
              <h3>Annotated Image:</h3>
              <img src={`http://localhost:5000/${annotatedImagePath}`} alt="Annotated pose" />
            </div>
          )}
        </div>

        {pose && (
          <div className="predicted-pose">
            <h3>Predicted Pose: {pose}</h3>
          </div>
        )}
      </div>
    );

}
