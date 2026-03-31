# Rice Quality Prediction API - Quick Guide

## Overview

The `/predict` endpoint accepts an image of rice grains and returns comprehensive quality analysis including:

- 15 predicted numerical characteristics
- Grain counts and measurements
- Defect analysis
- Quality classification (Premium, Good, Medium, Fair, Poor)

## Model Information

- **Model Used**: ONNX model (`Final_Best_model.onnx`)
- **Input**: 224×224 RGB image
- **Output**: 15 numerical predictions
- **Preprocessing**: ResNet/ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## API Endpoint

```
POST /predict
Authorization: Bearer <your_token>
Content-Type: multipart/form-data
```

### Request

Upload a rice grain image as a multipart form data file.

**Example using cURL:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -F "file=@path/to/rice_image.png"
```

**Example using Python:**

```python
import requests

url = "http://localhost:8000/predict"
headers = {"Authorization": f"Bearer {token}"}
files = {"file": open("rice_image.png", "rb")}

response = requests.post(url, headers=headers, files=files)
result = response.json()

print(f"Quality: {result['conclusion']['overall_quality_category']}")
print(f"Broken: {result['conclusion']['broken_grain_percentage']}%")
print(f"Defects: {result['conclusion']['defective_grain_percentage']}%")
```

## Response Structure

The API returns a JSON response with three main sections:

### 1. Sample Information

```json
{
  "sample_information": {
    "sample_id": "RICE_20260310_143022",
    "scan_id": "65f1234567890abcdef12345",
    "image_url": "https://res.cloudinary.com/.../scan.png",
    "scanned_at": "2026-03-10T14:30:22.123456"
  }
}
```

- **sample_id**: Unique identifier for the sample
- **scan_id**: MongoDB document ID for retrieving scan details later
- **image_url**: Permanent Cloudinary URL to the uploaded image
- **scanned_at**: ISO 8601 timestamp

### 2. Predicted Grain Characteristics

#### Grain Counts

```json
{
  "grain_characteristics": {
    "total_grains": 1523.45,
    "broken_grains": 123.82,
    "long_grains": 856.32,
    "medium_grains": 12.45
  }
}
```

#### Defective Grains

```json
{
  "defective_grains": {
    "black_grains": 34.67,
    "chalky_grains": 45.23,
    "red_grains": 23.45,
    "yellow_grains": 17.89,
    "green_grains": 12.34,
    "total_defective": 133.58
  }
}
```

#### Physical Measurements

```json
{
  "grain_measurements": {
    "average_length": 6.45,
    "average_width": 2.13,
    "length_width_ratio": 3.028
  }
}
```

Units: millimeters (mm)

#### Color Characteristics (LAB Color Space)

```json
{
  "color_characteristics": {
    "average_L": 65.23,
    "average_a": 5.67,
    "average_b": 23.45
  }
}
```

- **L**: Lightness (0=black, 100=white)
- **a**: Green(-) to Red(+) axis
- **b**: Blue(-) to Yellow(+) axis

### 3. Conclusion

```json
{
  "conclusion": {
    "broken_grain_percentage": 8.13,
    "defective_grain_percentage": 8.77,
    "overall_quality_category": "Good Quality",
    "quality_description": "Broken grains between 5% and 15%. Low defective grains. Good quality suitable for standard markets."
  }
}
```

## Quality Classification Rules

The system classifies rice into 5 categories based on calculated percentages:

### Premium Quality

- **Broken grains**: < 5%
- **Defective grains**: < 3%
- **Description**: Very low defects, uniform grain size and color
- **Market**: Premium/export grade

### Good Quality

- **Broken grains**: 5% - 15%
- **Defective grains**: < 8%
- **Description**: Low defects, good for standard markets
- **Market**: Standard retail

### Medium Quality

- **Broken grains**: 15% - 25%
- **Defective grains**: < 15%
- **Description**: Moderate defects, acceptable for general consumption
- **Market**: General consumption

### Fair Quality

- **Broken grains**: 25% - 35%
- **Defective grains**: < 25%
- **Description**: High defects, lower grade
- **Market**: Budget/institutional

### Poor Quality

- **Broken grains**: > 35% OR defective grains > 25%
- **Description**: Very high defects, irregular characteristics
- **Market**: Processing/animal feed only

## Calculated Indicators

The API automatically calculates:

### Broken Grain Percentage

```
Broken % = (Broken Count / Total Count) × 100
```

### Defective Grain Percentage

```
Defective % = (Black + Chalky + Red + Yellow + Green) / Total Count × 100
```

## Model Outputs Explained

| Output              | Description               | Use Case                              |
| ------------------- | ------------------------- | ------------------------------------- |
| Count               | Total grains detected     | Quantity assessment                   |
| Broken_Count        | Broken/damaged grains     | Quality indicator, milling efficiency |
| Long_Count          | Long grain variety count  | Variety classification                |
| Medium_Count        | Medium grain count        | Variety classification                |
| Black_Count         | Dark/diseased grains      | Quality defect                        |
| Chalky_Count        | Chalky/opaque grains      | Milling yield indicator               |
| Red_Count           | Red-colored grains        | Varietal purity                       |
| Yellow_Count        | Yellow/aged grains        | Storage condition indicator           |
| Green_Count         | Immature grains           | Harvest timing indicator              |
| WK_Length_Average   | Average grain length (mm) | Size classification                   |
| WK_Width_Average    | Average grain width (mm)  | Size classification                   |
| WK_LW_Ratio_Average | Length to width ratio     | Shape classification                  |
| Average_L           | Lightness value           | Appearance quality                    |
| Average_a           | Green-Red color value     | Color uniformity                      |
| Average_b           | Blue-Yellow color value   | Color uniformity                      |

## Error Handling

### 503 Service Unavailable

```json
{
  "detail": "Prediction model is not available"
}
```

**Cause**: Model failed to load on startup

### 400 Bad Request

```json
{
  "detail": "File must be an image (JPEG, PNG)"
}
```

**Cause**: Uploaded file is not a valid image

### 401 Unauthorized

```json
{
  "detail": "Could not validate credentials"
}
```

**Cause**: Invalid or missing authentication token

### 500 Internal Server Error

```json
{
  "detail": "Error processing image: <error_message>"
}
```

**Cause**: Processing error (check image format/size)

## Best Practices

1. **Image Requirements**
   - Format: JPEG or PNG
   - Size: 224×224 recommended (will be resized automatically)
   - Quality: Clear, well-lit images for best results
   - Content: Rice grains on contrasting background

2. **Authentication**
   - Always include valid JWT token in Authorization header
   - Token expires after 7 days by default

3. **Response Handling**
   - Store `scan_id` for retrieving results later
   - Use `image_url` for displaying the analyzed image
   - Cache quality category for quick status display

4. **Performance**
   - Model inference takes ~1-3 seconds per image
   - Images are uploaded to Cloudinary (may add latency)
   - Consider batching if analyzing multiple samples

## Testing

Use the included test script:

```bash
python test_api.py
```

This will test all endpoints including prediction if test images are available.

## Next Steps

- Retrieve scan history: `GET /scans`
- View specific scan: `GET /scans/{scan_id}`
- Delete old scans: `DELETE /scans/{scan_id}`

## Support

For issues or questions:

1. Check health endpoint: `GET /health`
2. Review API logs for detailed error messages
3. Ensure model file exists at `Saved_model/Final_Best_model.onnx`
