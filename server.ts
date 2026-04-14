import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: '50mb' }));

  // --- MOCK DATABASE & PIPELINE ---
  const detectionHistory: any[] = [];

  app.post("/api/predict", (req, res) => {
    const { image, lat, lng } = req.body;

    // Simulate Multi-Stage Pipeline Latency
    const stages = [
      { name: "Preprocessing", duration: 200 },
      { name: "CNN Feature Extraction", duration: 400 },
      { name: "YOLOv8 Localization", duration: 600 },
      { name: "EfficientNet Classification", duration: 800 }
    ];

    const diseases = [
      { name: "Potato___Late_blight", severity: "High", treatment: "CRITICAL: Destroy infected plants. Apply copper-based fungicides.", confidence: 0.96 },
      { name: "Tomato___Early_blight", severity: "Medium", treatment: "Improve air circulation, remove infected leaves", confidence: 0.92 },
      { name: "Corn_(maize)___Common_rust_", severity: "Low", treatment: "Plant resistant hybrids", confidence: 0.89 },
      { name: "Unsupported Crop (Mango) - Corrected by Hybrid AI", severity: "None", treatment: "The local model (specialized) misidentified this as Potato, but the Hybrid AI Expert (Gemini) corrected it to Mango. This crop is currently unsupported for disease diagnosis.", confidence: 0.45 }
    ];

    const result = diseases[Math.floor(Math.random() * diseases.length)];
    
    const detection = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      disease: result.name,
      severity: result.severity,
      treatment: result.treatment,
      confidence: result.confidence,
      is_unsupported: result.name.includes("Unsupported"),
      stages,
      location: {
        lat: lat || 18.5204 + (Math.random() - 0.5) * 0.02,
        lng: lng || 73.8567 + (Math.random() - 0.5) * 0.02
      }
    };

    detectionHistory.push(detection);
    res.json(detection);
  });

  app.get("/api/history", (req, res) => {
    res.json(detectionHistory);
  });

  // --- VITE MIDDLEWARE ---
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`GrowTogether-AI Live Prototype running on http://localhost:${PORT}`);
  });
}

startServer();
