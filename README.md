
# Open-Vakgyata JavaScript Implementation

JavaScript implementation of the [Open-Vakgyata](https://huggingface.co/onecxi/open-vakgyata) Indian language detection model using `Transformers.js` and ONNX runtime.

## 🎯 Overview

This script detects and classifies spoken Indian languages from 16kHz WAV audio using a quantized ONNX model of Wav2Vec2.

## 🌐 Supported Languages

- English (India) `en-IN`
- Hindi `hi-IN`
- Odia `or-IN`
- Bengali `bn-IN`
- Tamil `ta-IN`
- Telugu `te-IN`
- Kannada `kn-IN`
- Malayalam `ml-IN`
- Marathi `mr-IN`
- Gujarati `gu-IN`

## 🛠 Installation

```bash
npm install @xenova/transformers wav-decoder
```

## 🎧 Audio Requirements

- Format: WAV (Mono, 16-bit PCM)
- Sampling Rate: 16,000 Hz

To convert MP3:
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 sample.wav
```

## 📝 Usage

```bash
node index.js
```

## 📊 Output

```json
{
  "language": "hi-IN",
  "confidence": 0.9234,
  "allScores": [0.1, 0.9, ...]
}
```

## 📂 Files

- `index.js`: JavaScript inference script
- `sample.wav`: Input file (you must provide this)

---

Created by Aditya Mohla · Powered by Hugging Face + ONNX + Transformers.js
