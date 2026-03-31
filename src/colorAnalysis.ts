export function getClothingColorName(r: number, g: number, b: number): string {
  // Normalize brightness if it's too dark to detect hue accurately
  let rNorm = r / 255;
  let gNorm = g / 255;
  let bNorm = b / 255;
  const maxVal = Math.max(rNorm, gNorm, bNorm);
  
  // Boost colors if they are very dim
  if (maxVal > 0 && maxVal < 0.4) {
    const scale = 0.4 / maxVal;
    rNorm = Math.min(1, rNorm * scale);
    gNorm = Math.min(1, gNorm * scale);
    bNorm = Math.min(1, bNorm * scale);
  }
  
  const max = Math.max(rNorm, gNorm, bNorm);
  const min = Math.min(rNorm, gNorm, bNorm);
  let h = 0, s = 0, l = (max + min) / 2;

  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    switch (max) {
      case rNorm: h = (gNorm - bNorm) / d + (gNorm < bNorm ? 6 : 0); break;
      case gNorm: h = (bNorm - rNorm) / d + 2; break;
      case bNorm: h = (rNorm - gNorm) / d + 4; break;
    }
    h /= 6;
  }

  const hue = h * 360;
  const saturation = s * 100;
  const lightness = l * 100;

  // Black, White, Gray
  // If it's extremely dark, it's black
  if (lightness < 8) return '黑色'; 
  // If it's extremely bright and desaturated, it's white
  if (lightness > 85 && saturation < 15) return '白色';
  // If it's desaturated, it's gray or black depending on lightness
  if (saturation < 12) {
    return lightness < 30 ? '黑色' : '灰色';
  }

  // Colors based on Hue
  if (hue < 15 || hue >= 330) {
    if (saturation < 30 && lightness < 30) return '黑色';
    return hue >= 330 && saturation > 40 ? '粉色' : '红色';
  }
  if (hue < 45) return '橙色';
  if (hue < 75) return '黄色';
  if (hue < 160) return '绿色';
  if (hue < 260) return '蓝色';
  if (hue < 300) return '紫色';
  if (hue < 330) return '粉色';

  return '灰色';
}

export function isWarmColor(colorName: string): boolean {
  const warmColors = ['红色', '橙色', '黄色', '粉色'];
  return warmColors.includes(colorName);
}

export function isNeutralColor(colorName: string): boolean {
  const neutralColors = ['黑色', '白色', '灰色'];
  return neutralColors.includes(colorName);
}

export function isColdColor(colorName: string): boolean {
  const coldColors = ['蓝色', '绿色', '紫色'];
  return coldColors.includes(colorName);
}

export type ColorCategory = 'warm' | 'neutral' | 'cold' | 'other';

export function getColorCategory(colorName: string): ColorCategory {
  if (isWarmColor(colorName)) return 'warm';
  if (isNeutralColor(colorName)) return 'neutral';
  if (isColdColor(colorName)) return 'cold';
  return 'other';
}

export function sampleRegionColor(ctx: CanvasRenderingContext2D, x: number, y: number, width: number, height: number): { r: number, g: number, b: number } {
  try {
    const imageData = ctx.getImageData(x, y, width, height);
    const data = imageData.data;
    let rSum = 0, gSum = 0, bSum = 0;
    let totalWeight = 0;

    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      
      // Calculate saturation and brightness for weighting
      const max = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      const chroma = max - min;
      const saturation = max === 0 ? 0 : chroma / max;
      
      // Weighting strategy:
      // 1. Favor saturated pixels (they represent the "true" color better than shadows)
      // 2. Favor mid-to-high brightness pixels (avoid deep shadows)
      // 3. Ignore pure black or pure white noise
      const weight = (saturation * 2 + 0.1) * (max / 255 + 0.1);
      
      rSum += r * weight;
      gSum += g * weight;
      bSum += b * weight;
      totalWeight += weight;
    }

    if (totalWeight === 0) return { r: 128, g: 128, b: 128 };

    return {
      r: Math.round(rSum / totalWeight),
      g: Math.round(gSum / totalWeight),
      b: Math.round(bSum / totalWeight)
    };
  } catch (e) {
    console.error("Error sampling color:", e);
    return { r: 128, g: 128, b: 128 };
  }
}
