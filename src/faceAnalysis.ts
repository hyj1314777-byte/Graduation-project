import { FaceShape, Emotion, FaceMacroCategory, FaceAnalysisResult } from './types';

/**
 * Categorize face shapes into macro-groups for calligraphy mapping.
 */
const getMacroCategory = (shape: FaceShape): FaceMacroCategory => {
  if (shape === '椭圆形脸型' || shape === '圆脸型') return '圆润光滑型';
  if (shape === '长脸型' || shape === '方脸型') return '方正直线型';
  if (shape === '倒三角脸型' || shape === '正三角脸型' || shape === '菱形脸型') return '几何突变型';
  return '检测中...';
};

/**
 * Geometric analysis of face shape based on MediaPipe landmarks.
 */

export function analyzeEmotion(landmarks: any[]): Emotion {
  if (!landmarks || landmarks.length === 0) return '检测中...';

  const getDist = (p1: any, p2: any) => {
    return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
  };

  // 关键点获取
  const mouthL = landmarks[61], mouthR = landmarks[291]; // 嘴角
  const lipTop = landmarks[13], lipBottom = landmarks[14]; // 上下唇
  const eyeLTop = landmarks[159], eyeLBot = landmarks[145]; // 左眼上下
  const eyeRTop = landmarks[386], eyeRBot = landmarks[374]; // 右眼上下
  const browLInner = landmarks[52], browRInner = landmarks[282]; // 眉毛内侧
  const noseTip = landmarks[1];
  const mouthCenterY = (lipTop.y + lipBottom.y) / 2;

  // 计算特征值
  const mouthWidth = getDist(mouthL, mouthR);
  const mouthOpen = getDist(lipTop, lipBottom);
  const eyeOpenL = getDist(eyeLTop, eyeLBot);
  const eyeOpenR = getDist(eyeRTop, eyeRBot);
  const avgEyeOpen = (eyeOpenL + eyeOpenR) / 2;
  
  // 归一化参考值
  const faceWidth = getDist(landmarks[234], landmarks[454]);
  const faceHeight = getDist(landmarks[10], landmarks[152]);
  const normMouthWidth = mouthWidth / faceWidth;
  const normMouthOpen = mouthOpen / faceHeight;
  const normEyeOpen = avgEyeOpen / faceHeight;

  // 嘴角上扬/下垂程度 (负值表示上扬，正值表示下垂)
  const mouthCornerLift = ((mouthL.y - mouthCenterY) + (mouthR.y - mouthCenterY)) / 2 / faceHeight;

  // 1. 高兴 (Happiness): 优先级最高，防止微笑被误判为恐惧或惊讶
  // 核心特征：嘴角明显上扬
  if (mouthCornerLift < -0.008 || (normMouthWidth > 0.30 && mouthCornerLift < 0)) {
    return '高兴';
  }

  // 2. 惊讶 (Surprise): 眼睛明显瞪大 + 嘴巴张开 (O型)
  // 增加阈值并排除微笑干扰
  if (normEyeOpen > 0.042 && normMouthOpen > 0.07 && mouthCornerLift > -0.002) {
    return '惊讶';
  }

  // 3. 愤怒 (Anger): 极大提升灵敏度，响应“稍微皱眉”的需求
  const browInnerDist = getDist(browLInner, browRInner) / faceWidth;
  const eyeBrowDistL = getDist(browLInner, eyeLTop);
  const eyeBrowDistR = getDist(browRInner, eyeRTop);
  const normEyeBrowDist = (eyeBrowDistL + eyeBrowDistR) / 2 / faceHeight;
  const noseLipDist = getDist(noseTip, lipTop) / faceHeight;
  
  // 判定条件 1: 轻微皱眉 (阈值进一步放宽)
  const isFrowning = browInnerDist < 0.33;
  // 判定条件 2: 轻微眉毛下压 (阈值进一步放宽)
  const isBrowLowered = normEyeBrowDist < 0.11;
  // 判定条件 3: 怒吼/张嘴
  const isShouting = normMouthOpen > 0.05 && browInnerDist < 0.35;
  // 判定条件 4: 狰狞/皱鼻
  const isGrimacing = noseLipDist < 0.06 && browInnerDist < 0.35;
  // 判定条件 5: 眯眼 + 皱眉 (愤怒的典型特征)
  const isSquinting = normEyeOpen < 0.022;

  // 综合判定：只要不是在明显微笑，满足以上任意路径即判定为愤怒
  // 调低了对嘴角上扬的排斥阈值，从 -0.002 降至 -0.005，允许轻微的嘴角上扬（有些人的自然唇形）
  if (mouthCornerLift > -0.005) {
    if (isFrowning || isBrowLowered || isShouting || isGrimacing || (isSquinting && isFrowning)) return '愤怒';
  }

  // 4. 难过 (Sadness): 嘴角下垂
  if (mouthCornerLift > 0.012) {
    return '难过';
  }

  return '平静';
}

export function analyzeFaceShape(landmarks: any[], hairlineOffset: number = 0.15): FaceAnalysisResult {
  if (!landmarks || landmarks.length === 0) {
    return { 
      shape: '检测中...', 
      macroCategory: '检测中...',
      emotion: '检测中...', 
      confidence: 0, 
      measurements: {
        ratioLenWidth: 0,
        ratioForeheadCheek: 0,
        ratioJawCheek: 0
      } 
    };
  }

  const emotion = analyzeEmotion(landmarks);
  const getDist = (p1: any, p2: any) => {
    return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
  };

  // 1. 获取核心测量点
  // L: 额头顶(10) 到 下巴底(152)
  const pTopRaw = landmarks[10];
  const pBottomRaw = landmarks[152];
  const vX = pTopRaw.x - pBottomRaw.x;
  const vY = pTopRaw.y - pBottomRaw.y;

  // 使用传入的动态发际线补偿值
  const L_base = getDist(pTopRaw, pBottomRaw);
  const L = L_base * (1 + hairlineOffset); 

  // FB (额头宽度) 恢复到原始锚点位置
  const FB = getDist(landmarks[103], landmarks[332]);

  const CB = getDist(landmarks[234], landmarks[454]);
  const MB = getDist(landmarks[132], landmarks[361]);
  const maxW = Math.max(FB, CB, MB);

  // 2. 定义辅助判断变量 (容差 5%)
  const T = 0.05;
  const fb_cb_eq = Math.abs(FB - CB) / Math.max(FB, CB) < T;
  const cb_mb_eq = Math.abs(CB - MB) / Math.max(CB, MB) < T;
  const all_eq = fb_cb_eq && cb_mb_eq;

  let shape: FaceShape = '椭圆形脸型';
  let confidence = 0.85;

  // 3. 核心算法逻辑 (按优先级排序)
  
  // A. 长脸型 (Long): MAX(FB, CB, MB) < 2/3 * L
  if (maxW < (2 / 3) * L) {
    shape = '长脸型';
  }
  // B. 倒三角脸型 (Inverted Triangle): FB > CB > MB
  else if (FB > CB * (1 + T) && CB > MB * (1 + T)) {
    shape = '倒三角脸型';
  }
  // C. 正三角脸型 (Triangle): FB < CB < MB
  else if (FB < CB * (1 - T) && CB < MB * (1 - T)) {
    shape = '正三角脸型';
  }
  // D. 菱形脸型 (Diamond): FB < CB 且 CB > MB
  else if (FB < CB * (1 - T) && CB > MB * (1 + T)) {
    shape = '菱形脸型';
  }
  // E. 圆脸型 vs 方脸型 (宽度基本相等)
  else if (all_eq) {
    // 圆脸型: FB=CB=MB ≈ L (长宽比接近 1)
    if (Math.abs(maxW - L) / Math.max(maxW, L) < 0.15) {
      shape = '圆脸型';
    } 
    // 方脸型: FB=CB=MB <= L
    else {
      shape = '方脸型';
    }
  }
  // F. 椭圆形脸型 (Oval): FB = CB > MB
  else if (fb_cb_eq && CB > MB * (1 + T)) {
    shape = '椭圆形脸型';
  }
  // G. 兜底逻辑
  else {
    shape = '椭圆形脸型';
  }

  return {
    shape,
    macroCategory: getMacroCategory(shape),
    emotion,
    confidence,
    measurements: {
      ratioLenWidth: parseFloat((L / CB).toFixed(2)),
      ratioForeheadCheek: parseFloat((FB / CB).toFixed(2)),
      ratioJawCheek: parseFloat((MB / CB).toFixed(2)),
      L: parseFloat(L.toFixed(4)),
      FB: parseFloat(FB.toFixed(4)),
      CB: parseFloat(CB.toFixed(4)),
      MB: parseFloat(MB.toFixed(4)),
      landmarks: landmarks
    }
  };
}
