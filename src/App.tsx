import React, { useEffect, useRef, useState } from 'react';
import { FaceMesh } from '@mediapipe/face_mesh';
import * as cam from '@mediapipe/camera_utils';
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';
import { FACEMESH_TESSELATION, FACEMESH_RIGHT_EYE, FACEMESH_LEFT_EYE, FACEMESH_FACE_OVAL, FACEMESH_LIPS, FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYEBROW } from '@mediapipe/face_mesh';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Camera, 
  RefreshCw, 
  Activity, 
  User, 
  LayoutGrid, 
  Info, 
  BrainCircuit, 
  ChevronRight, 
  Maximize2,
  Cpu,
  Scan
} from 'lucide-react';
import { cn, FaceShape, Emotion, FaceAnalysisResult } from './types';
import { analyzeFaceShape } from './faceAnalysis';
import { getClothingColorName, sampleRegionColor, getColorCategory, ColorCategory, isWarmColor, isNeutralColor, isColdColor } from './colorAnalysis';
import { GoogleGenAI } from "@google/genai";

declare global {
  interface Window {
    aistudio: {
      hasSelectedApiKey: () => Promise<boolean>;
      openSelectKey: () => Promise<void>;
    };
  }
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sampleCanvasRef = useRef<HTMLCanvasElement>(document.createElement('canvas'));
  const [result, setResult] = useState<FaceAnalysisResult | null>(null);
  const resultBuffer = useRef<FaceShape[]>([]);
  const emotionBuffer = useRef<Emotion[]>([]);
  const colorBuffer = useRef<string[]>([]);
  const rgbBuffer = useRef<{r: number, g: number, b: number}[]>([]);
  const hairlineOffsetRef = useRef<number>(0.12); // 初始值
  const BUFFER_SIZE = 3; 
  const [isAnalyzing, setIsAnalyzing] = useState(true);

  const [isCameraReady, setIsCameraReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isGeneratingArt, setIsGeneratingArt] = useState(false);
  const [artImageUrl, setArtImageUrl] = useState<string | null>(null);
  const [artError, setArtError] = useState<string | null>(null);

  const detectHairline = (video: HTMLVideoElement, topPoint: {x: number, y: number}, faceHeight: number) => {
    const sampleCanvas = sampleCanvasRef.current;
    const ctx = sampleCanvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return 0.12;

    const stripWidth = 10;
    const stripHeight = Math.floor(faceHeight * 0.3 * video.videoHeight); // 搜索范围为脸高的30%
    if (stripHeight <= 5) return 0.12;

    const centerX = topPoint.x * video.videoWidth;
    const centerY = topPoint.y * video.videoHeight;

    sampleCanvas.width = stripWidth;
    sampleCanvas.height = stripHeight;

    // 绘制额头上方区域
    ctx.drawImage(
      video,
      Math.max(0, centerX - stripWidth / 2), Math.max(0, centerY - stripHeight), 
      stripWidth, stripHeight,
      0, 0, stripWidth, stripHeight
    );

    const imageData = ctx.getImageData(0, 0, stripWidth, stripHeight);
    const data = imageData.data;

    let maxGradient = 0;
    let bestY = stripHeight;

    // 从下往上搜索最强亮度梯度 (皮肤到头发的转换)
    for (let y = stripHeight - 2; y > 2; y--) {
      let bCurr = 0, bAbove = 0;
      for (let x = 0; x < stripWidth; x++) {
        const i = (y * stripWidth + x) * 4;
        const iA = ((y - 1) * stripWidth + x) * 4;
        bCurr += (data[i] + data[i+1] + data[i+2]) / 3;
        bAbove += (data[iA] + data[iA+1] + data[iA+2]) / 3;
      }
      const gradient = Math.abs(bCurr - bAbove);
      if (gradient > maxGradient) {
        maxGradient = gradient;
        bestY = y;
      }
    }

    const detectedOffset = (stripHeight - bestY) / video.videoHeight;
    // 限制在合理范围内 (5% - 25%) 并进行平滑处理
    const clampedOffset = Math.max(0.05, Math.min(0.25, detectedOffset));
    hairlineOffsetRef.current = hairlineOffsetRef.current * 0.8 + clampedOffset * 0.2;
    return hairlineOffsetRef.current;
  };

  useEffect(() => {
    const faceMesh = new FaceMesh({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
      },
    });

    faceMesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    faceMesh.onResults((results) => {
      if (!canvasRef.current || !videoRef.current) return;

      const canvasCtx = canvasRef.current.getContext('2d');
      if (!canvasCtx) return;

      canvasCtx.save();
      canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      
      // Draw the image to the canvas first so we can sample from it
      if (results.image) {
        canvasCtx.drawImage(results.image, 0, 0, canvasRef.current.width, canvasRef.current.height);
      }
      
      if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0 && isAnalyzing) {
        const landmarks = results.multiFaceLandmarks[0];
        
        // 1. Sample clothing color (below chin) BEFORE drawing landmarks
        const chin = landmarks[152];
        const sampleW = Math.min(320, 0.5 * canvasRef.current.width);
        const sampleH_max = Math.min(80, 0.15 * canvasRef.current.height);
        
        const sampleX = Math.max(0, Math.min(canvasRef.current.width - sampleW, (chin.x * canvasRef.current.width) - (sampleW / 2)));
        const sampleY = Math.min(canvasRef.current.height - 40, (chin.y + 0.12) * canvasRef.current.height);
        const sampleH = Math.min(sampleH_max, canvasRef.current.height - sampleY);
        
        let colorName = '灰色';
        let colorCategory: ColorCategory = 'neutral';
        let sampledRgb = { r: 128, g: 128, b: 128 };
        try {
          sampledRgb = sampleRegionColor(canvasCtx, sampleX, sampleY, sampleW, sampleH);
          colorName = getClothingColorName(sampledRgb.r, sampledRgb.g, sampledRgb.b);
          colorCategory = getColorCategory(colorName);
        } catch (e) {
          console.error("Color sampling failed", e);
        }

        // 2. Clear the image to restore the high-tech overlay look
        canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

        // 3. Draw detailed landmarks for "precision" look
        canvasCtx.globalAlpha = 0.15;
        drawConnectors(canvasCtx, landmarks, FACEMESH_TESSELATION, { color: '#10b981', lineWidth: 0.5 });
        
        canvasCtx.globalAlpha = 0.6;
        drawConnectors(canvasCtx, landmarks, FACEMESH_LIPS, { color: '#10b981', lineWidth: 1 });
        drawConnectors(canvasCtx, landmarks, FACEMESH_LEFT_EYE, { color: '#10b981', lineWidth: 1 });
        drawConnectors(canvasCtx, landmarks, FACEMESH_RIGHT_EYE, { color: '#10b981', lineWidth: 1 });
        drawConnectors(canvasCtx, landmarks, FACEMESH_LEFT_EYEBROW, { color: '#10b981', lineWidth: 1 });
        drawConnectors(canvasCtx, landmarks, FACEMESH_RIGHT_EYEBROW, { color: '#10b981', lineWidth: 1 });
        
        // Draw Iris manually for high-tech feel
        canvasCtx.globalAlpha = 0.8;
        const irisIndices = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477];
        irisIndices.forEach(idx => {
          const p = landmarks[idx];
          if (p) {
            canvasCtx.beginPath();
            canvasCtx.arc(p.x * canvasRef.current!.width, p.y * canvasRef.current!.height, 1, 0, 2 * Math.PI);
            canvasCtx.fillStyle = '#34d399';
            canvasCtx.fill();
          }
        });
        
        // Draw tracking box
        const xCoords = landmarks.map(l => l.x * canvasRef.current!.width);
        const yCoords = landmarks.map(l => l.y * canvasRef.current!.height);
        const minX = Math.min(...xCoords), maxX = Math.max(...xCoords);
        const minY = Math.min(...yCoords), maxY = Math.max(...yCoords);
        const padding = 20;
        
        canvasCtx.strokeStyle = 'rgba(16, 185, 129, 0.4)';
        canvasCtx.lineWidth = 1;
        canvasCtx.strokeRect(minX - padding, minY - padding, (maxX - minX) + padding * 2, (maxY - minY) + padding * 2);
        
        // Draw small corner markers for the box
        const box = { x: minX - padding, y: minY - padding, w: (maxX - minX) + padding * 2, h: (maxY - minY) + padding * 2 };
        canvasCtx.fillStyle = '#10b981';
        canvasCtx.fillRect(box.x, box.y, 10, 2);
        canvasCtx.fillRect(box.x, box.y, 2, 10);
        canvasCtx.fillRect(box.x + box.w - 10, box.y, 10, 2);
        canvasCtx.fillRect(box.x + box.w - 2, box.y, 2, 10);
        canvasCtx.fillRect(box.x, box.y + box.h - 2, 10, 2);
        canvasCtx.fillRect(box.x, box.y + box.h - 10, 2, 10);
        canvasCtx.fillRect(box.x + box.w - 10, box.y + box.h - 2, 10, 2);
        canvasCtx.fillRect(box.x + box.w - 2, box.y + box.h - 10, 2, 10);

        // Draw dynamic coordinates label
        canvasCtx.fillStyle = 'rgba(16, 185, 129, 0.8)';
        canvasCtx.font = '10px monospace';
        canvasCtx.fillText(`X: ${minX.toFixed(0)} Y: ${minY.toFixed(0)}`, box.x, box.y - 5);
        canvasCtx.fillText(`W: ${box.w.toFixed(0)} H: ${box.h.toFixed(0)}`, box.x + box.w - 60, box.y - 5);

        // 1. 获取核心测量点并计算平滑发际线向量
        const pTopRaw = landmarks[10];
        const pBottomRaw = landmarks[152];
        const vX = pTopRaw.x - pBottomRaw.x;
        const vY = pTopRaw.y - pBottomRaw.y;
        
        // 调用实时像素级发际线检测技术
        const faceHeight = Math.sqrt(Math.pow(vX, 2) + Math.pow(vY, 2));
        const dynamicHairlineOffset = detectHairline(videoRef.current, pTopRaw, faceHeight);

        // 2. 绘制平滑覆盖发际线的面部轮廓
        canvasCtx.beginPath();
        canvasCtx.strokeStyle = 'rgba(16, 185, 129, 0.8)';
        canvasCtx.lineWidth = 2;
        
        // 全脸轮廓点位序列
        const ovalIndices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10];
        
        // 定义额头区域的加权提升系数 (正态分布，确保圆滑)
        const foreheadWeights: Record<number, number> = {
          10: dynamicHairlineOffset,   // 顶点：使用动态检测值
          338: dynamicHairlineOffset * 0.8,  // 侧上方：平滑过渡
          109: dynamicHairlineOffset * 0.8,
          297: dynamicHairlineOffset * 0.5,  // 侧边：微调
          67: dynamicHairlineOffset * 0.5,
          332: dynamicHairlineOffset * 0.1,  // 锚点：几乎不提升，确保与侧脸接轨
          103: dynamicHairlineOffset * 0.1
        };

        ovalIndices.forEach((index, i) => {
          const p = landmarks[index];
          let drawX = p.x * canvasRef.current!.width;
          let drawY = p.y * canvasRef.current!.height;
          
          // 应用平滑提升算法
          if (foreheadWeights[index] !== undefined) {
            const weight = foreheadWeights[index];
            drawX = (p.x + vX * weight) * canvasRef.current!.width;
            drawY = (p.y + vY * weight) * canvasRef.current!.height;
          }

          if (i === 0) canvasCtx.moveTo(drawX, drawY);
          else canvasCtx.lineTo(drawX, drawY);
        });
        canvasCtx.stroke();

        // 3. 绘制中轴线 (L线)，同步至平滑发际线顶点
        const startX = (pTopRaw.x + vX * dynamicHairlineOffset) * canvasRef.current.width;
        const startY = (pTopRaw.y + vY * dynamicHairlineOffset) * canvasRef.current.height;
        const endX = pBottomRaw.x * canvasRef.current.width;
        const endY = pBottomRaw.y * canvasRef.current.height;

        canvasCtx.beginPath();
        canvasCtx.setLineDash([5, 5]);
        canvasCtx.moveTo(startX, startY);
        canvasCtx.lineTo(endX, endY);
        canvasCtx.strokeStyle = 'rgba(16, 185, 129, 0.9)';
        canvasCtx.lineWidth = 2;
        canvasCtx.stroke();
        
        // Draw FB (Forehead Width) line: 103 to 332 (Reverted to original anatomical position)
        canvasCtx.beginPath();
        canvasCtx.moveTo(landmarks[103].x * canvasRef.current.width, landmarks[103].y * canvasRef.current.height);
        canvasCtx.lineTo(landmarks[332].x * canvasRef.current.width, landmarks[332].y * canvasRef.current.height);
        canvasCtx.strokeStyle = 'rgba(52, 211, 153, 0.6)';
        canvasCtx.stroke();

        // Draw CB (Cheekbone Width) line: 234 to 454
        canvasCtx.beginPath();
        canvasCtx.moveTo(landmarks[234].x * canvasRef.current.width, landmarks[234].y * canvasRef.current.height);
        canvasCtx.lineTo(landmarks[454].x * canvasRef.current.width, landmarks[454].y * canvasRef.current.height);
        canvasCtx.strokeStyle = 'rgba(52, 211, 153, 0.6)';
        canvasCtx.stroke();

        // Draw MB (Mandible Width) line: 132 to 361
        canvasCtx.beginPath();
        canvasCtx.moveTo(landmarks[132].x * canvasRef.current.width, landmarks[132].y * canvasRef.current.height);
        canvasCtx.lineTo(landmarks[361].x * canvasRef.current.width, landmarks[361].y * canvasRef.current.height);
        canvasCtx.strokeStyle = 'rgba(52, 211, 153, 0.6)';
        canvasCtx.stroke();
        
        canvasCtx.setLineDash([]);

        // Add labels for lines
        canvasCtx.fillStyle = '#34d399';
        canvasCtx.font = 'bold 10px monospace';
        canvasCtx.fillText('L', startX, startY - 10);
        canvasCtx.fillText('FB', landmarks[103].x * canvasRef.current.width - 20, landmarks[103].y * canvasRef.current.height);
        canvasCtx.fillText('CB', landmarks[234].x * canvasRef.current.width - 20, landmarks[234].y * canvasRef.current.height);
        canvasCtx.fillText('MB', landmarks[132].x * canvasRef.current.width - 20, landmarks[132].y * canvasRef.current.height);

        // Draw key points as small dots for precision feel
        canvasCtx.globalAlpha = 0.4;
        drawLandmarks(canvasCtx, landmarks, {
          color: '#10b981',
          lineWidth: 0,
          radius: 0.5,
        });
        
        // Analyze face shape & emotion
        const analysis = analyzeFaceShape(landmarks, dynamicHairlineOffset);
        
        // Draw sample area indicator
        const categoryColors = {
          warm: '#f87171',
          neutral: '#9ca3af',
          cold: '#60a5fa',
          other: '#34d399'
        };
        canvasCtx.strokeStyle = categoryColors[colorCategory];
        canvasCtx.lineWidth = 1;
        canvasCtx.strokeRect(sampleX, sampleY, sampleW, sampleH);

        // 平滑处理逻辑 (脸型)
        resultBuffer.current.push(analysis.shape);
        if (resultBuffer.current.length > BUFFER_SIZE) {
          resultBuffer.current.shift();
        }

        // 平滑处理逻辑 (情绪)
        emotionBuffer.current.push(analysis.emotion);
        if (emotionBuffer.current.length > BUFFER_SIZE) {
          emotionBuffer.current.shift();
        }

        // 平滑处理逻辑 (颜色)
        colorBuffer.current.push(colorName);
        if (colorBuffer.current.length > BUFFER_SIZE) {
          colorBuffer.current.shift();
        }

        rgbBuffer.current.push(sampledRgb);
        if (rgbBuffer.current.length > BUFFER_SIZE) {
          rgbBuffer.current.shift();
        }

        const counts: Record<string, number> = {};
        resultBuffer.current.forEach(s => counts[s] = (counts[s] || 0) + 1);
        const smoothedShape = Object.entries(counts).reduce((a, b) => a[1] > b[1] ? a : b)[0] as FaceShape;

        const eCounts: Record<string, number> = {};
        emotionBuffer.current.forEach(e => eCounts[e] = (eCounts[e] || 0) + 1);
        const smoothedEmotion = Object.entries(eCounts).reduce((a, b) => a[1] > b[1] ? a : b)[0] as Emotion;

        const cCounts: Record<string, number> = {};
        colorBuffer.current.forEach(c => cCounts[c] = (cCounts[c] || 0) + 1);
        const smoothedColorName = Object.entries(cCounts).reduce((a, b) => a[1] > b[1] ? a : b)[0] as string;

        const smoothedRgb = rgbBuffer.current.reduce((acc, curr) => ({
          r: acc.r + curr.r / rgbBuffer.current.length,
          g: acc.g + curr.g / rgbBuffer.current.length,
          b: acc.b + curr.b / rgbBuffer.current.length
        }), { r: 0, g: 0, b: 0 });

        setResult({
          ...analysis,
          shape: smoothedShape,
          emotion: smoothedEmotion,
          clothingColor: smoothedColorName,
          colorCategory: getColorCategory(smoothedColorName),
          sampledRgb: {
            r: Math.round(smoothedRgb.r),
            g: Math.round(smoothedRgb.g),
            b: Math.round(smoothedRgb.b)
          }
        });
      } else if (!isAnalyzing) {
        setResult(null);
      }
      canvasCtx.restore();
    });

    if (videoRef.current) {
      const camera = new cam.Camera(videoRef.current, {
        onFrame: async () => {
          if (videoRef.current) {
            await faceMesh.send({ image: videoRef.current });
          }
        },
        width: 640,
        height: 480,
      });
      camera.start()
        .then(() => setIsCameraReady(true))
        .catch((err) => {
          console.error('Camera initialization error:', err);
          if (err.name === 'NotAllowedError' || err.message?.includes('denied')) {
            setError('PERMISSION_DENIED');
          } else {
            setError('无法访问摄像头，请检查设备连接或尝试刷新页面。');
          }
        });
    }

    return () => {
      faceMesh.close();
    };
  }, []);

  const handleRefresh = () => {
    resultBuffer.current = [];
    emotionBuffer.current = [];
    setResult(null);
    setArtImageUrl(null);
    setIsAnalyzing(true);
  };

  const toggleAnalysis = () => {
    setIsAnalyzing(!isAnalyzing);
    if (!isAnalyzing) {
      handleRefresh();
    }
  };

  const getShapeDescription = (shape: FaceShape) => {
    switch (shape) {
      case '倒三角脸型': return '额头宽，下巴尖，线条流畅，是东方审美的经典脸型。';
      case '圆脸型': return '面部丰满，长宽相近，线条柔和，显得亲切活泼。';
      case '方脸型': return '额头、颧骨、下颌宽度相近，轮廓分明，极具力量感。';
      case '长脸型': return '脸部长度明显大于宽度，通常额头或下巴较长，显得成熟稳重。';
      case '椭圆形脸型': return '比例均匀，线条圆润，是最理想的标准脸型。';
      case '正三角脸型': return '额头窄，下颌宽，重心偏下，显得沉稳厚重。';
      case '菱形脸型': return '颧骨最宽，额头和下巴较窄，轮廓感强，极具个性。';
      default: return '请正对摄像头，保持光线充足。';
    }
  };

  const generateArt = async () => {
    if (!result) return;
    setArtError(null);

    setIsGeneratingArt(true);
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
      
      const getCalligraphyPrompt = (category: string, emotion: string, colorCat: ColorCategory) => {
        let baseStyle = "";
        let emotionDetail = "";
        
        if (category === '圆润光滑型') {
          baseStyle = "风格参考：秦篆与王羲之行书。特征：藏锋入笔、圆转连绵、线条粗细均匀且温润。";
          switch(emotion) {
            case '高兴': emotionDetail = "墨韵丰盈，行云流水，线条连绵不断，笔触饱满，带有温暖的轻微晕染。"; break;
            case '难过': emotionDetail = "水墨氤氲，如泣如诉，墨色湿润泛晕，笔意迟滞，边缘模糊，透着哀伤。"; break;
            case '惊讶': emotionDetail = "顿笔圆润，气韵外扩，犹如浓墨坠入清水瞬间漾开，带有弹性的扩张感。"; break;
            case '愤怒': emotionDetail = "藏锋暗劲，重墨如磐，下笔极重，墨色浓黑压抑，张力内敛于浑厚笔画中。"; break;
            default: emotionDetail = "气韵匀净，静水流深，线条恬淡自然，粗细均匀，心如止水。";
          }
        } else if (category === '方正直线型') {
          baseStyle = "风格参考：唐楷与魏碑。特征：方折顿挫、骨力遒劲、法度严谨、金石气、结构稳定。";
          switch(emotion) {
            case '高兴': emotionDetail = "舒展大开，明朗堂正，大开大合，如斧凿刀刻般明朗，自信大气，线条充满向上的张力。"; break;
            case '难过': emotionDetail = "沉郁顿挫，枯墨重按，笔断意连，线条带有极强滞涩感，如苍松承雪，透着厚重的压抑感。"; break;
            case '惊讶': emotionDetail = "起笔突兀，方折猛烈，线条转折处带有极强物理冲击力，犹如金石崩裂，结构瞬间炸裂。"; break;
            case '愤怒': emotionDetail = "力透纸背，斩钉截铁，浓墨与渴笔交织，字形如怒目金刚，极具爆发力，笔触狂暴且带有极端的视觉张力。"; break;
            default: emotionDetail = "结体端庄，稳如泰山，法度森严，呈现雕塑般的庄严感。";
          }
        } else { // 几何突变型
          baseStyle = "风格参考：宋徽宗瘦金体与张旭狂草。特征：露锋直入、笔锋凌厉、险绝峭拔、宽窄对比极端。";
          switch(emotion) {
            case '高兴': emotionDetail = "飞白跳跃，笔锋轻灵，线条灵动锐利，犹如燕子掠水，带有干脆的飞白，节奏欢快跳跃。"; break;
            case '难过': emotionDetail = "瘦骨清相，游丝断裂，笔锋尖锐凄冷，线条极细如牵丝，带有破碎美感，如寒蝉凄切。"; break;
            case '惊讶': emotionDetail = "险绝破局，偏锋刺入，线条呈锐角放射，多用偏锋，犹如闪电划破夜空，视觉冲击极强。"; break;
            case '愤怒': emotionDetail = "笔锋凌厉，铁画银钩，狂草走势，满纸烟云，字字如刀剑出鞘，锋芒锐利，线条如狂风暴雨般席卷画面。"; break;
            default: emotionDetail = "锋芒内敛，寒梅傲雪，结体清瘦孤傲，透着疏离感与冷峻。";
          }
        }

        let colorInstruction = "";
        switch (colorCat) {
          case 'warm':
            colorInstruction = "【极其重要：覆盖性视觉特征】请务必打破基础风格的纤细感，转而使用极其浑厚的线条，字势向外大幅度扩张，墨色饱满，呈现出颜真卿《大麻姑仙坛记》那种雄强、宽博、厚重的气韵。";
            break;
          case 'neutral':
            colorInstruction = "【核心视觉特征】具备“稳”且“净”的质感，每一个笔画极其讲究规矩，法度严谨，结体方整，呈现出一种端庄肃穆、纤尘不染的秩序美。";
            break;
          case 'cold':
            colorInstruction = "【核心视觉特征】具备“乱”、“燥”、“真”的质感。画面中应出现涂抹、修改的痕迹，顾不得排版，运笔极快，带有大量“渴笔”和“枯墨”，展现出一种心情激愤、不计工拙的真实情感爆发。";
            break;
        }

        return `${baseStyle} 情绪表现：${emotionDetail} ${colorInstruction}`;
      };

      const calligraphyContext = getCalligraphyPrompt(result.macroCategory, result.emotion, result.colorCategory || 'other');

      const prompt = `
        你是一位研究未来AI时代语言系统的艺术家。
        请根据以下面部分析结果，创作一幅“解构主义书法”作品。
        这件作品应呈现一种“似字非字”的神秘美感，保留汉字的骨架神韵，但无法被识别为具体文字。
        
        【体验者特征】
        - 脸型类别：${result.macroCategory} (${result.shape})
        - 当前情绪：${result.emotion}
        
        【艺术指导要求】
        1. 核心视觉：${calligraphyContext}
        2. 结构神韵：模仿汉字的“间架结构”与“骨干”，呈现出一种“伪文字”(Pseudo-logographic)的形态。线条之间应有呼应、避让和重心的平衡感，仿佛是某种失传的古老文字或未来天书。
        3. 似是而非：严禁生成任何标准的、可阅读的汉字。它应该是汉字被拆解、重组、异化后的视觉残留，保留笔画的起承转合，达到“似像又不像”的境界。
        4. 构图与意境：强调中国书法的空间留白（计白当黑）。展现出一种由生物信号驱动的、具有灵魂厚度的视觉符号。
        5. 媒介与背景：模拟宣纸上的水墨效果，包含飞白、晕染、浓淡、焦墨等传统技法。**必须确保背景为纯白色 (#FFFFFF)，没有任何杂质、阴影或纸张纹理，以便与网页背景完美融合。**
        
        请直接生成这幅背景纯白的、具有“似像又不像”美感的解构主义书法艺术作品。
      `;

      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash-image',
        contents: { parts: [{ text: prompt }] },
        config: {
          imageConfig: {
            aspectRatio: "1:1"
          }
        }
      });

      if (response.candidates?.[0]?.content?.parts) {
        for (const part of response.candidates[0].content.parts) {
          if (part.inlineData) {
            setArtImageUrl(`data:image/png;base64,${part.inlineData.data}`);
            setArtError(null);
            break;
          }
        }
      } else {
        setArtError('生成失败：未返回图像数据');
      }
    } catch (error: any) {
      console.error('Art generation failed', error);
      setArtError(`生成失败: ${error.message || '未知错误'}`);
    } finally {
      setIsGeneratingArt(false);
    }
  };

  return (
    <div className="min-h-screen bg-white text-black p-4 md:p-8 font-sans selection:bg-emerald-500/30">
      {/* Header */}
      <header className="max-w-7xl mx-auto mb-8 flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <motion.div 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center gap-2 text-emerald-500 mb-2"
          >
            <Scan size={20} />
            <span className="text-xs font-mono uppercase tracking-widest">Biometric Analysis System v1.0</span>
          </motion.div>
          <motion.h1 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="text-4xl md:text-6xl font-bold tracking-tighter"
          >
            Algorithmic Script <span className="text-emerald-500 italic">Allocation</span>
          </motion.h1>
        </div>
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="flex items-center gap-3"
        >
          <button 
            onClick={toggleAnalysis}
            className={cn(
              "flex items-center gap-2 px-6 py-3 rounded-2xl font-mono text-xs uppercase tracking-widest transition-all active:scale-95 shadow-lg shadow-emerald-500/10",
              isAnalyzing 
                ? "bg-red-500/20 border border-red-500/30 text-red-400 hover:bg-red-500/30" 
                : "bg-emerald-500/20 border border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/30"
            )}
          >
            {isAnalyzing ? (
              <><Activity size={16} className="animate-pulse" /> Stop Analysis</>
            ) : (
              <><Scan size={16} /> Start Analysis</>
            )}
          </button>

          <button 
            onClick={handleRefresh}
            className="flex items-center gap-2 px-6 py-3 rounded-2xl bg-black/5 border border-black/10 text-black/80 font-mono text-xs uppercase tracking-widest hover:bg-black/10 hover:text-black transition-all active:scale-95 group shadow-lg"
          >
            <RefreshCw size={16} className="group-hover:rotate-180 transition-transform duration-500" />
            Refresh / Reset
          </button>

          <div className="hidden xl:flex items-center gap-4 text-xs font-mono text-black/40 ml-4">
            <div className="flex items-center gap-2">
              <div className={cn("w-2 h-2 rounded-full animate-pulse", isCameraReady ? "bg-emerald-500" : "bg-red-500")} />
              {isCameraReady ? "SYSTEM ONLINE" : "SYSTEM INITIALIZING"}
            </div>
          </div>
        </motion.div>
      </header>

      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Camera Section */}
        <div className="lg:col-span-8 space-y-4">
          <div className="relative aspect-[4/3] rounded-3xl overflow-hidden glass-panel group">
            {!isCameraReady && !error && (
              <div className="absolute inset-0 flex flex-col items-center justify-center z-20 bg-white/60 backdrop-blur-md">
                <RefreshCw className="animate-spin text-emerald-500 mb-4" size={40} />
                <p className="font-mono text-sm animate-pulse text-black">INITIALIZING CAMERA...</p>
              </div>
            )}
            
            {error && (
              <div className="absolute inset-0 flex flex-col items-center justify-center z-20 bg-red-50/80 backdrop-blur-xl border border-red-500/30 p-8 text-center">
                <div className="w-16 h-16 bg-red-500/20 rounded-full flex items-center justify-center mb-6">
                  <Info className="text-red-500" size={32} />
                </div>
                <h3 className="text-xl font-bold text-black mb-2">
                  {error === 'PERMISSION_DENIED' ? '摄像头权限未开启' : '摄像头启动失败'}
                </h3>
                <p className="text-sm text-black/60 mb-8 max-w-md leading-relaxed">
                  {error === 'PERMISSION_DENIED' 
                    ? (
                      <>
                        系统无法获取摄像头权限。这通常是因为浏览器禁用了权限请求。<br/><br/>
                        <span className="text-emerald-400 font-bold">解决方法：</span><br/>
                        1. 点击浏览器地址栏左侧的 <span className="underline">锁头图标</span>，将“摄像头”设置为“允许”。<br/>
                        2. 如果您在预览窗口中，请尝试点击右上角的 <span className="underline">“在新标签页中打开”</span> 图标，在独立页面中授权。<br/>
                        3. 确保没有其他程序正在占用摄像头。
                      </>
                    )
                    : error}
                </p>
                <button 
                  onClick={() => window.location.reload()}
                  className="px-8 py-3 bg-red-500 hover:bg-red-600 text-white rounded-2xl font-mono text-xs uppercase tracking-widest transition-all active:scale-95 shadow-lg shadow-red-500/20"
                >
                  刷新页面并重试
                </button>
              </div>
            )}
 
            <video
              ref={videoRef}
              className="absolute inset-0 w-full h-full object-contain brightness-90 contrast-110"
              playsInline
              muted
            />
            
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full z-10 pointer-events-none"
              width={640}
              height={480}
            />

            {/* Scanning Overlay */}
            <div className="absolute inset-0 pointer-events-none border-[20px] border-white/20 z-10" />
            <div className="scanner-line" />
            
            {/* Corner Accents */}
            <div className="absolute top-8 left-8 w-12 h-12 border-t-2 border-l-2 border-emerald-500 z-20" />
            <div className="absolute top-8 right-8 w-12 h-12 border-t-2 border-r-2 border-emerald-500 z-20" />
            <div className="absolute bottom-8 left-8 w-12 h-12 border-b-2 border-l-2 border-emerald-500 z-20" />
            <div className="absolute bottom-8 right-8 w-12 h-12 border-b-2 border-r-2 border-emerald-500 z-20" />

            {/* HUD Labels */}
            <div className="absolute top-12 left-12 font-mono text-[10px] text-emerald-500/60 z-20 uppercase tracking-widest">
              Tracking Active
            </div>
            <div className="absolute bottom-12 right-12 font-mono text-[10px] text-emerald-500/60 z-20 uppercase tracking-widest">
              60 FPS / 1080P
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div className="glass-panel p-4 rounded-2xl flex items-center gap-3">
              <div className="p-2 bg-emerald-500/10 rounded-lg text-emerald-500">
                <Activity size={18} />
              </div>
              <div>
                <p className="text-[10px] text-black/40 uppercase font-mono">Confidence</p>
                <p className="font-mono text-sm text-black">{result ? `${(result.confidence * 100).toFixed(0)}%` : '--'}</p>
              </div>
            </div>
            <div className="glass-panel p-4 rounded-2xl flex items-center gap-3">
              <div className="p-2 bg-emerald-500/10 rounded-lg text-emerald-500">
                <User size={18} />
              </div>
              <div>
                <p className="text-[10px] text-black/40 uppercase font-mono">Face Detected</p>
                <p className="font-mono text-sm text-black">{result ? 'YES' : 'NO'}</p>
              </div>
            </div>
            <div className="glass-panel p-4 rounded-2xl flex items-center gap-3">
              <div className="p-2 bg-emerald-500/10 rounded-lg text-emerald-500">
                <LayoutGrid size={18} />
              </div>
              <div>
                <p className="text-[10px] text-black/40 uppercase font-mono">Landmarks</p>
                <p className="font-mono text-sm text-black">478 PTS</p>
              </div>
            </div>
          </div>
        </div>

        {/* Info Section */}
        <aside className="lg:col-span-4 space-y-6">
          <section className="glass-panel p-6 rounded-3xl border-black/10 bg-black/[0.02]">
            <h2 className="text-xs font-mono text-emerald-600 uppercase tracking-[0.2em] mb-4">Color Recognition</h2>
            <AnimatePresence mode="wait">
              {result?.clothingColor ? (
                <motion.div
                  key={result.clothingColor}
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  className="flex items-center gap-4"
                >
                  <div 
                    className="w-12 h-12 rounded-2xl shadow-inner border border-black/10"
                    style={{ 
                      backgroundColor: result.sampledRgb ? `rgb(${result.sampledRgb.r}, ${result.sampledRgb.g}, ${result.sampledRgb.b})` : 
                                       result.colorCategory === 'warm' ? '#f87171' : 
                                       result.colorCategory === 'cold' ? '#60a5fa' : 
                                       result.colorCategory === 'neutral' ? '#9ca3af' : '#34d399' 
                    }}
                  />
                  <div>
                    <p className="text-[10px] font-mono text-black/40 uppercase tracking-widest">Detected Color</p>
                    <div className="flex items-center gap-2">
                      <span className="text-xl font-bold tracking-tighter text-black">{result.clothingColor}</span>
                      <span className={cn(
                        "text-[10px] font-mono px-2 py-0.5 rounded border",
                        result.colorCategory === 'warm' ? "bg-red-500/10 border-red-500/20 text-red-400" :
                        result.colorCategory === 'cold' ? "bg-blue-500/10 border-blue-500/20 text-blue-400" :
                        "bg-gray-500/10 border-gray-500/20 text-gray-400"
                      )}>
                        {result.colorCategory?.toUpperCase()}
                      </span>
                    </div>
                  </div>
                </motion.div>
              ) : (
                <p className="text-[10px] font-mono text-black/20 italic">Waiting for color sampling...</p>
              )}
            </AnimatePresence>
          </section>

          <section className="glass-panel p-6 rounded-3xl border-black/10 bg-black/[0.02]">
            <h2 className="text-xs font-mono text-emerald-600 uppercase tracking-[0.2em] mb-4">Analysis Result</h2>
            
            <AnimatePresence mode="wait">
              {result ? (
                <motion.div
                  key={result.shape + result.emotion}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="space-y-6"
                >
                  <div className="space-y-1">
                    <p className="text-[10px] font-mono text-black/40 uppercase tracking-widest">Face Result</p>
                    <div className="flex items-baseline gap-2">
                      <span className="text-4xl font-bold tracking-tighter text-black">{result.macroCategory}</span>
                      <span className="text-xs font-mono text-emerald-500 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                        {result.shape}
                      </span>
                    </div>
                    <p className="text-black/60 leading-relaxed text-xs">
                      {getShapeDescription(result.shape)}
                    </p>
                  </div>

                  <div className="pt-4 border-t border-black/15 space-y-1">
                    <p className="text-[10px] font-mono text-black/40 uppercase tracking-widest">Emotion Result</p>
                    <div className="flex items-center gap-3">
                      <span className="text-3xl font-bold tracking-tighter text-emerald-500">{result.emotion}</span>
                      <div className="px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20 text-[10px] font-mono text-emerald-500">
                        LIVE_FEED
                      </div>
                    </div>
                  </div>

                  <div className="pt-4 border-t border-black/15 space-y-4">
                    <p className="text-[10px] font-mono text-black/40 uppercase tracking-widest">Abstract Face Art</p>
                    {artImageUrl ? (
                      <motion.div 
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="relative aspect-square rounded-2xl overflow-hidden border border-black/10 bg-white group"
                      >
                        <img 
                          src={artImageUrl} 
                          alt="Abstract Face Art" 
                          className="w-full h-full object-cover mix-blend-multiply"
                          referrerPolicy="no-referrer"
                        />
                        <button 
                          onClick={() => setArtImageUrl(null)}
                          className="absolute top-2 right-2 p-1.5 bg-black/60 backdrop-blur-md rounded-lg text-white/60 hover:text-white opacity-0 group-hover:opacity-100 transition-opacity"
                        >
                          <RefreshCw size={14} />
                        </button>
                      </motion.div>
                    ) : (
                      <div className="space-y-2">
                        <button
                          onClick={generateArt}
                          disabled={isGeneratingArt}
                          className="w-full py-4 rounded-2xl border border-emerald-500/20 bg-emerald-500/5 hover:bg-emerald-500/10 transition-all flex flex-col items-center justify-center gap-2 group disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          {isGeneratingArt ? (
                            <>
                              <RefreshCw size={20} className="animate-spin text-emerald-500" />
                              <span className="text-[10px] font-mono text-emerald-500 uppercase tracking-widest animate-pulse">Generating Abstract Art...</span>
                            </>
                          ) : (
                            <>
                              <div className="p-2 bg-emerald-500/10 rounded-xl text-emerald-500 group-hover:scale-110 transition-transform">
                                <BrainCircuit size={20} />
                              </div>
                              <span className="text-[10px] font-mono text-emerald-500 uppercase tracking-widest">Generate Abstract Glyph</span>
                            </>
                          )}
                        </button>
                        {artError && (
                          <p className="text-[10px] font-mono text-red-400 text-center animate-pulse">{artError}</p>
                        )}
                      </div>
                    )}
                  </div>
                </motion.div>
              ) : (
                <div className="space-y-4 py-8 flex flex-col items-center justify-center text-center">
                  <div className="w-12 h-12 border-2 border-emerald-500/20 border-t-emerald-500 rounded-full animate-spin mb-4" />
                  <p className="text-black/40 font-mono text-xs uppercase tracking-widest">Waiting for face detection...</p>
                </div>
              )}
            </AnimatePresence>
          </section>

          <section className="glass-panel p-6 rounded-3xl space-y-6 border-black/10 bg-black/[0.02]">
            <h2 className="text-xs font-mono text-black/40 uppercase tracking-[0.2em]">Geometric Metrics</h2>
            
            <div className="space-y-4">
              <MetricRow 
                label="Length/Width Ratio" 
                value={result?.measurements.ratioLenWidth || 0} 
                target="1.0 - 1.6"
              />
              <MetricRow 
                label="Forehead/Cheek Ratio" 
                value={result?.measurements.ratioForeheadCheek || 0} 
                target="0.8 - 1.1"
              />
              <MetricRow 
                label="Jaw/Cheek Ratio" 
                value={result?.measurements.ratioJawCheek || 0} 
                target="0.6 - 0.9"
              />
            </div>
          </section>

          <div className="p-6 rounded-3xl bg-black/5 border border-black/10">
            <div className="flex items-start gap-3">
              <Info size={16} className="text-emerald-500 shrink-0 mt-0.5" />
              <p className="text-[11px] text-black/40 leading-relaxed">
                识别结果基于面部关键点比例计算，受光照、角度及表情影响。建议在光线充足的环境下，保持面部正对摄像头以获得最准确的分析。
              </p>
            </div>
          </div>
        </aside>
      </main>
      
      <footer className="max-w-7xl mx-auto mt-12 pt-8 border-t border-black/5 flex justify-between items-center text-[10px] font-mono text-black/20 uppercase tracking-widest">
        <div>© 2026 ALGORITHMIC SCRIPT ALLOCATION</div>
        <div className="flex gap-6">
          <a href="#" className="hover:text-emerald-500 transition-colors">Privacy</a>
          <a href="#" className="hover:text-emerald-500 transition-colors">Terms</a>
          <a href="#" className="hover:text-emerald-500 transition-colors">API Docs</a>
        </div>
      </footer>
    </div>
  );
}

function MetricRow({ label, value, target }: { label: string; value: number; target: string }) {
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between text-[10px] font-mono uppercase tracking-wider">
        <span className="text-black/40">{label}</span>
        <span className="text-emerald-500">{value.toFixed(2)}</span>
      </div>
      <div className="h-1 bg-black/5 rounded-full overflow-hidden">
        <motion.div 
          className="h-full bg-emerald-500"
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(value * 50, 100)}%` }}
        />
      </div>
      <div className="flex justify-between text-[9px] font-mono text-black/20">
        <span>REF: {target}</span>
      </div>
    </div>
  );
}
