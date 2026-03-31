import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export interface FaceLandmarks {
  [key: string]: { x: number; y: number; z: number };
}

export type FaceShape = 
  | '长脸型' 
  | '倒三角脸型' 
  | '正三角脸型' 
  | '菱形脸型' 
  | '圆脸型' 
  | '方脸型' 
  | '椭圆形脸型'
  | '检测中...';

export type Emotion = 
  | '高兴' 
  | '难过' 
  | '愤怒' 
  | '惊讶' 
  | '平静'
  | '检测中...';

export type FaceMacroCategory = '圆润光滑型' | '方正直线型' | '几何突变型' | '检测中...';

export interface FaceAnalysisResult {
  shape: FaceShape;
  macroCategory: FaceMacroCategory;
  emotion: Emotion;
  clothingColor?: string;
  colorCategory?: 'warm' | 'neutral' | 'cold' | 'other';
  sampledRgb?: { r: number, g: number, b: number };
  confidence: number;
  measurements: {
    ratioLenWidth: number;
    ratioForeheadCheek: number;
    ratioJawCheek: number;
    L?: number;
    FB?: number;
    CB?: number;
    MB?: number;
    landmarks?: any[];
  };
}
