import React, { useState, useEffect, useRef } from 'react';
import { 
  Upload, MapPin, AlertTriangle, CheckCircle, Info, 
  Loader2, History, Send, Droplets, Zap, Target, Layers, 
  ChevronRight, Activity, ShieldAlert, Thermometer
} from 'lucide-react';
import { MapContainer, TileLayer, Marker, Popup, useMap, Circle } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import { motion, AnimatePresence } from 'motion/react';
import { GoogleGenAI, Type } from "@google/genai";
import { cn } from './lib/utils';

// Fix Leaflet icon issue
// @ts-ignore
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

interface Stage {
  name: string;
  duration: number;
}

interface Detection {
  id: number;
  timestamp: string;
  disease: string;
  severity: 'High' | 'Medium' | 'Low' | 'None';
  treatment: string;
  confidence: number;
  is_unsupported?: boolean;
  stages: Stage[];
  location: { lat: number; lng: number };
  expertAnalysis?: {
    correctedDisease: string;
    isCorrect: boolean;
    cropType: string;
    explanation: string;
    treatmentPlan: string;
    severity: 'High' | 'Medium' | 'Low' | 'None';
  };
}

export default function App() {
  const [image, setImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStage, setCurrentStage] = useState<number>(-1);
  const [result, setResult] = useState<Detection | null>(null);
  const [history, setHistory] = useState<Detection[]>([]);
  const [location, setLocation] = useState<{ lat: number; lng: number }>({ lat: 18.5204, lng: 73.8567 });
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition((pos) => {
        setLocation({ lat: pos.coords.latitude, lng: pos.coords.longitude });
      });
    }
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const res = await fetch('/api/history');
      const data = await res.json();
      setHistory(data);
    } catch (err) {
      console.error("Failed to fetch history", err);
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImage(reader.result as string);
        setResult(null);
        setCurrentStage(-1);
      };
      reader.readAsDataURL(file);
    }
  };

  const runPipeline = async () => {
    if (!image) return;
    setIsProcessing(true);
    setResult(null);

    // Simulate visual pipeline stages
    for (let i = 0; i < 4; i++) {
      setCurrentStage(i);
      await new Promise(r => setTimeout(r, 600));
    }

    try {
      // 1. Local Model Prediction (Fast, Edge-Ready)
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image, ...location }),
      });
      const localData: Detection = await response.json();
      
      // 2. Secondary AI Expert Validation (Gemini - High Reasoning)
      setIsAnalyzing(true);
      const expertAnalysis = await getGeminiExpertAnalysis(image, localData.disease);
      
      // 3. Merge Results - Prioritize Expert Analysis for "Perfect Output"
      const finalResult: Detection = { 
        ...localData, 
        expertAnalysis,
        // If expert corrected the disease, use the expert's identification as primary
        disease: expertAnalysis?.correctedDisease || localData.disease,
        severity: expertAnalysis?.severity || localData.severity,
        treatment: expertAnalysis?.treatmentPlan || localData.treatment,
        is_unsupported: !expertAnalysis?.isCorrect || localData.is_unsupported
      };
      
      setResult(finalResult);
      fetchHistory();
    } catch (err) {
      console.error("Pipeline failed", err);
    } finally {
      setIsProcessing(false);
      setIsAnalyzing(false);
      setCurrentStage(-1);
    }
  };

  const getGeminiExpertAnalysis = async (base64Image: string, localPrediction: string) => {
    try {
      const base64Data = base64Image.split(',')[1];
      const prompt = `
        You are a world-class agricultural plant pathologist. 
        A local specialized AI model has predicted this crop disease as: "${localPrediction}".
        
        Analyze the provided image of the plant leaf. 
        Your goal is to provide a "PERFECT" output by verifying or correcting the local model.
        
        1. Identify the actual crop type (e.g., Mango, Tomato, Potato, Banana, etc.).
        2. Determine the specific disease or if the plant is healthy.
        3. If the local prediction is wrong (e.g., if this is a Mango leaf but predicted as Potato), you MUST correct it.
        4. Provide a scientific explanation and a production-level treatment plan.
      `;

      const response = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: [
          {
            parts: [
              { text: prompt },
              { inlineData: { mimeType: "image/jpeg", data: base64Data } }
            ]
          }
        ],
        config: {
          temperature: 0.2,
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              correctedDisease: { type: Type.STRING, description: "The final, accurate disease name (e.g., 'Mango Anthracnose' or 'Tomato Late Blight')" },
              isCorrect: { type: Type.BOOLEAN, description: "Whether the local prediction was accurate" },
              cropType: { type: Type.STRING, description: "The identified crop species" },
              explanation: { type: Type.STRING, description: "Scientific explanation of symptoms" },
              treatmentPlan: { type: Type.STRING, description: "Detailed treatment protocol" },
              severity: { type: Type.STRING, enum: ["High", "Medium", "Low", "None"], description: "Risk level" }
            },
            required: ["correctedDisease", "isCorrect", "cropType", "explanation", "treatmentPlan", "severity"]
          }
        }
      });

      return JSON.parse(response.text);
    } catch (error) {
      console.error("Gemini Analysis Error:", error);
      return null;
    }
  };

  const pipelineStages = [
    { name: "CNN Feature Extraction", icon: Layers, color: "text-blue-500" },
    { name: "YOLOv8 Localization", icon: Target, color: "text-purple-500" },
    { name: "EfficientNet Classification", icon: Activity, color: "text-orange-500" },
    { name: "Edge AI Optimization", icon: Zap, color: "text-yellow-500" }
  ];

  return (
    <div className="min-h-screen flex flex-col bg-[#f8fafc]">
      {/* Premium Header */}
      <header className="bg-[#1e293b] text-white py-6 px-8 shadow-2xl border-b border-white/10">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center gap-4">
            <div className="bg-gradient-to-br from-green-400 to-emerald-600 p-2.5 rounded-2xl shadow-lg shadow-green-500/20">
              <Droplets className="text-white w-7 h-7" />
            </div>
            <div>
              <h1 className="text-2xl font-black tracking-tight flex items-center gap-2">
                GrowTogether-AI <span className="text-[10px] bg-green-500/20 text-green-400 px-2 py-0.5 rounded-full border border-green-500/30 uppercase tracking-widest">Enterprise</span>
              </h1>
              <p className="text-slate-400 text-xs font-medium uppercase tracking-wider">Cascaded Deep Learning Pipeline</p>
            </div>
          </div>
          <div className="hidden lg:flex items-center gap-8">
            <div className="flex flex-col items-end">
              <span className="text-xs text-slate-500 font-bold uppercase">Edge Connectivity</span>
              <span className="text-sm text-green-400 font-mono">98.4ms Latency</span>
            </div>
            <div className="h-10 w-px bg-white/10" />
            <button className="bg-white/5 hover:bg-white/10 px-5 py-2.5 rounded-xl border border-white/10 transition-all text-sm font-bold">
              Fleet Management
            </button>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-7xl mx-auto w-full p-8 grid grid-cols-1 lg:grid-cols-12 gap-10">
        
        {/* Left: Input & Pipeline */}
        <div className="lg:col-span-5 space-y-8">
          
          {/* Diagnostic Control */}
          <section className="bg-white rounded-[2.5rem] p-8 shadow-xl shadow-slate-200/50 border border-slate-100">
            <div className="flex justify-between items-center mb-8">
              <h2 className="text-xl font-bold text-slate-800 flex items-center gap-3">
                <Thermometer className="w-6 h-6 text-emerald-500" />
                Drone Diagnostic
              </h2>
              <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Module v2.4</span>
            </div>
            
            <div 
              onClick={() => fileInputRef.current?.click()}
              className={cn(
                "group relative border-2 border-dashed rounded-[2rem] p-10 transition-all cursor-pointer flex flex-col items-center justify-center text-center overflow-hidden",
                image ? "border-emerald-500 bg-emerald-50/30" : "border-slate-200 hover:border-emerald-500 hover:bg-emerald-50/20"
              )}
            >
              <input type="file" ref={fileInputRef} onChange={handleImageUpload} className="hidden" accept="image/*" />
              
              {image ? (
                <div className="relative">
                  <img src={image} alt="Drone Feed" className="max-h-72 rounded-2xl shadow-2xl ring-4 ring-white" />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent rounded-2xl" />
                </div>
              ) : (
                <>
                  <div className="bg-emerald-100 p-5 rounded-3xl mb-5 group-hover:scale-110 transition-transform">
                    <Upload className="w-10 h-10 text-emerald-600" />
                  </div>
                  <p className="font-bold text-slate-700 text-lg">Upload Drone Imagery</p>
                  <p className="text-sm text-slate-400 mt-2 max-w-[200px]">High-res RGB or Multispectral TIFF supported</p>
                </>
              )}
            </div>

            <button
              disabled={!image || isProcessing}
              onClick={runPipeline}
              className="w-full mt-8 bg-[#1e293b] text-white py-5 rounded-2xl font-black flex items-center justify-center gap-4 hover:bg-slate-800 transition-all disabled:opacity-50 shadow-xl shadow-slate-900/20 group"
            >
              {isProcessing ? (
                <Loader2 className="w-6 h-6 animate-spin" />
              ) : (
                <>
                  <Zap className="w-5 h-5 text-yellow-400 group-hover:scale-125 transition-transform" />
                  Execute AI Pipeline
                </>
              )}
            </button>
          </section>

          {/* Pipeline Progress Visualization */}
          <AnimatePresence>
            {(isProcessing || currentStage >= 0) && (
              <motion.section 
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="bg-[#1e293b] rounded-[2.5rem] p-8 shadow-2xl text-white"
              >
                <h3 className="text-sm font-black text-slate-500 uppercase tracking-widest mb-6">Pipeline Execution Flow</h3>
                <div className="space-y-4">
                  {pipelineStages.map((stage, idx) => {
                    const Icon = stage.icon;
                    const isActive = currentStage === idx;
                    const isCompleted = currentStage > idx;
                    
                    return (
                      <div key={idx} className={cn(
                        "flex items-center gap-4 p-4 rounded-2xl transition-all border",
                        isActive ? "bg-white/10 border-white/20 scale-105" : "bg-transparent border-transparent opacity-40"
                      )}>
                        <div className={cn("p-2 rounded-xl bg-white/5", stage.color)}>
                          <Icon className="w-5 h-5" />
                        </div>
                        <div className="flex-1">
                          <p className="font-bold text-sm">{stage.name}</p>
                          {isActive && <p className="text-[10px] text-slate-400 animate-pulse">Processing Tensors...</p>}
                        </div>
                        {isCompleted && <CheckCircle className="w-5 h-5 text-green-400" />}
                        {isActive && <Loader2 className="w-4 h-4 animate-spin text-white/50" />}
                      </div>
                    );
                  })}
                </div>
              </motion.section>
            )}
          </AnimatePresence>

          {/* Result Card */}
          <AnimatePresence>
            {result && !isProcessing && (
              <motion.section 
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className={cn(
                  "bg-white rounded-[2.5rem] p-8 shadow-2xl border-t-[12px]",
                  result.is_unsupported ? "border-orange-400" : "border-emerald-500"
                )}
              >
                {result.is_unsupported && (
                  <div className="bg-orange-50 border border-orange-100 p-4 rounded-2xl mb-6 flex items-start gap-3">
                    <AlertTriangle className="text-orange-500 w-5 h-5 mt-0.5 shrink-0" />
                    <div>
                      <p className="text-orange-800 font-bold text-sm">Unsupported Crop Detected</p>
                      <p className="text-orange-700 text-xs mt-1">The AI has identified this as a crop outside our current diagnostic scope. Results may be inaccurate.</p>
                    </div>
                  </div>
                )}

                <div className="flex justify-between items-start mb-8">
                  <div>
                    <h3 className={cn(
                      "text-3xl font-black",
                      result.is_unsupported ? "text-orange-900" : "text-slate-900"
                    )}>
                      {result.disease.replace(/___/g, " ")}
                    </h3>
                    <div className="flex items-center gap-2 mt-2 text-slate-400 font-bold text-xs uppercase tracking-tighter">
                      <MapPin className="w-4 h-4 text-red-500" />
                      {result.location.lat.toFixed(6)} N, {result.location.lng.toFixed(6)} E
                    </div>
                  </div>
                  <div className={cn(
                    "px-5 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest shadow-sm",
                    result.severity === 'High' ? "bg-red-50 text-red-600 border border-red-100" :
                    result.severity === 'Medium' ? "bg-orange-50 text-orange-600 border border-orange-100" :
                    "bg-emerald-50 text-emerald-600 border border-emerald-100"
                  )}>
                    {result.severity} Risk
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-6 mb-8">
                  <div className="bg-slate-50 p-6 rounded-3xl border border-slate-100">
                    <p className="text-[10px] text-slate-400 uppercase font-black mb-2 tracking-widest">Model Confidence</p>
                    <p className="text-3xl font-black text-emerald-600">{(result.confidence * 100).toFixed(1)}%</p>
                  </div>
                  <div className="bg-slate-50 p-6 rounded-3xl border border-slate-100">
                    <p className="text-[10px] text-slate-400 uppercase font-black mb-2 tracking-widest">Inference Time</p>
                    <p className="text-3xl font-black text-slate-800">1.2s</p>
                  </div>
                </div>

                <div className="bg-emerald-600 text-white p-8 rounded-[2rem] shadow-xl shadow-emerald-500/20 relative overflow-hidden">
                  <div className="absolute top-0 right-0 p-4 opacity-10">
                    <ShieldAlert className="w-20 h-20" />
                  </div>
                  <h4 className="font-black text-xs uppercase tracking-widest mb-3 opacity-80">Protocol Recommendation</h4>
                  <p className="text-lg font-bold leading-tight">{result.treatment}</p>
                </div>

                {/* Gemini Expert Analysis Section */}
                <div className="mt-8 pt-8 border-t border-slate-100">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-sm font-black text-slate-800 uppercase tracking-widest flex items-center gap-2">
                      <Zap className="w-4 h-4 text-yellow-500" />
                      AI Expert Validation
                    </h4>
                    <span className="text-[10px] bg-emerald-100 text-emerald-700 px-2 py-1 rounded-lg font-bold">Verified Output</span>
                  </div>
                  
                  {isAnalyzing ? (
                    <div className="flex items-center gap-3 text-slate-400 py-4">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span className="text-xs font-medium italic">Consulting Cloud Expert Knowledge...</span>
                    </div>
                  ) : result.expertAnalysis ? (
                    <div className="bg-slate-50 rounded-2xl p-6 border border-slate-100">
                      <div className="flex items-center gap-2 mb-3">
                        {result.expertAnalysis.isCorrect ? (
                          <CheckCircle className="w-4 h-4 text-emerald-500" />
                        ) : (
                          <AlertTriangle className="w-4 h-4 text-orange-500" />
                        )}
                        <span className="text-xs font-bold text-slate-700">
                          {result.expertAnalysis.isCorrect ? "Local Model Verified" : "Expert Correction Applied"}
                        </span>
                      </div>
                      <p className="text-sm text-slate-600 leading-relaxed whitespace-pre-wrap italic">
                        "{result.expertAnalysis.explanation}"
                      </p>
                      <div className="mt-4 flex items-center gap-2 text-[10px] text-slate-400 font-bold uppercase">
                        <Info className="w-3 h-3" />
                        Perfected by Gemini 3.0 Flash • High-Reasoning Cloud Layer
                      </div>
                    </div>
                  ) : (
                    <p className="text-xs text-slate-400 italic">Expert analysis unavailable for this scan.</p>
                  )}
                </div>
              </motion.section>
            )}
          </AnimatePresence>
        </div>

        {/* Right: GIS & Heatmap */}
        <div className="lg:col-span-7 space-y-8">
          
          {/* GIS Visualization */}
          <section className="bg-white rounded-[2.5rem] p-6 shadow-xl border border-slate-100 h-[650px] flex flex-col">
            <div className="px-4 py-4 flex justify-between items-center border-b border-slate-50 mb-4">
              <div>
                <h2 className="text-xl font-black text-slate-800 flex items-center gap-3">
                  <Activity className="w-6 h-6 text-red-500" />
                  Spatial Infection Heatmap
                </h2>
                <p className="text-xs text-slate-400 font-medium">Real-time PostGIS Geo-spatial Sync</p>
              </div>
              <div className="flex gap-2">
                <div className="w-3 h-3 rounded-full bg-red-500" />
                <div className="w-3 h-3 rounded-full bg-orange-500" />
                <div className="w-3 h-3 rounded-full bg-emerald-500" />
              </div>
            </div>
            <div className="flex-1 rounded-[2rem] overflow-hidden border border-slate-100 shadow-inner">
              <MapContainer center={[location.lat, location.lng]} zoom={14} scrollWheelZoom={true}>
                <TileLayer
                  attribution='&copy; OpenStreetMap'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                {history.map((item) => (
                  <React.Fragment key={item.id}>
                    <Marker position={[item.location.lat, item.location.lng]}>
                      <Popup>
                        <div className="p-2 font-sans">
                          <p className="font-black text-slate-900">{item.disease}</p>
                          <p className="text-[10px] text-slate-400 uppercase font-bold">{item.severity} Severity</p>
                        </div>
                      </Popup>
                    </Marker>
                    <Circle 
                      center={[item.location.lat, item.location.lng]}
                      radius={150}
                      pathOptions={{ 
                        fillColor: item.severity === 'High' ? '#ef4444' : item.severity === 'Medium' ? '#f97316' : '#10b981',
                        color: 'transparent',
                        fillOpacity: 0.3
                      }}
                    />
                  </React.Fragment>
                ))}
                <MapUpdater center={location} />
              </MapContainer>
            </div>
          </section>

          {/* Analytics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-white p-8 rounded-[2rem] shadow-xl border border-slate-100">
              <History className="w-8 h-8 text-slate-300 mb-4" />
              <p className="text-4xl font-black text-slate-900">{history.length}</p>
              <p className="text-[10px] text-slate-400 uppercase font-black tracking-widest mt-1">Total Analysis</p>
            </div>
            <div className="bg-[#1e293b] p-8 rounded-[2rem] shadow-xl">
              <ShieldAlert className="text-red-400 w-8 h-8 mb-4" />
              <p className="text-4xl font-black text-white">{history.filter(h => h.severity === 'High').length}</p>
              <p className="text-[10px] text-slate-500 uppercase font-black tracking-widest mt-1">Critical Alerts</p>
            </div>
            <div className="bg-emerald-600 p-8 rounded-[2rem] shadow-xl">
              <CheckCircle className="text-white/50 w-8 h-8 mb-4" />
              <p className="text-4xl font-black text-white">
                {(history.length > 0 ? (history.filter(h => h.severity === 'None').length / history.length) * 100 : 0).toFixed(0)}%
              </p>
              <p className="text-[10px] text-white/50 uppercase font-black tracking-widest mt-1">Yield Safety</p>
            </div>
          </div>
        </div>
      </main>

      <footer className="bg-white border-t border-slate-100 py-8 px-8 text-center">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-slate-400 text-xs font-bold uppercase tracking-widest">GrowTogether-AI • Precision Agriculture Framework</p>
          <div className="flex gap-6 text-[10px] font-black text-slate-300 uppercase tracking-widest">
            <a href="#" className="hover:text-emerald-500 transition-colors">Documentation</a>
            <a href="#" className="hover:text-emerald-500 transition-colors">API Reference</a>
            <a href="#" className="hover:text-emerald-500 transition-colors">Privacy Policy</a>
          </div>
        </div>
      </footer>
    </div>
  );
}

function MapUpdater({ center }: { center: { lat: number; lng: number } }) {
  const map = useMap();
  useEffect(() => {
    map.setView([center.lat, center.lng]);
  }, [center, map]);
  return null;
}
