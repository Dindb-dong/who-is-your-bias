/* Classic worker: the MediaPipe WASM loader requires importScripts. All assets
 * are same-origin and pinned by scripts/prepare-vision.mjs. No frames leave this
 * worker, are retained, or are sent to the application server. */
let detector = null;
let initializing = false;
let lastTimestamp = -Infinity;
self.onmessage = async (event) => {
  const message = event.data;
  if (message.type === 'init') {
    if (initializing || detector) return;
    initializing = true;
    try {
      const { FaceDetector, FilesetResolver } = await import('./vision_bundle.mjs');
      const vision = await FilesetResolver.forVisionTasks(new URL('./wasm', self.location.href).href);
      detector = await FaceDetector.createFromOptions(vision, {
        baseOptions: { modelAssetPath: new URL('./blaze_face_short_range.tflite', self.location.href).href, delegate: 'CPU' },
        runningMode: 'VIDEO', minDetectionConfidence: 0.6, minSuppressionThreshold: 0.3,
      });
      self.postMessage({ type: 'ready' });
    } catch {
      self.postMessage({ type: 'error', message: 'The local face detector could not load. Run npm run prepare:vision, then restart the studio.' });
    }
    return;
  }
  if (message.type !== 'frame') return;
  const bitmap = message.bitmap;
  try {
    if (!detector || !Number.isFinite(message.observedAtMs) || message.observedAtMs <= lastTimestamp) throw new Error('Invalid or stale video frame.');
    lastTimestamp = message.observedAtMs;
    const result = detector.detectForVideo(bitmap, message.observedAtMs);
    const width = bitmap.width; const height = bitmap.height;
    const boxes = result.detections.slice(0, 8).flatMap((detection) => {
      const box = detection.boundingBox;
      if (!box || !width || !height) return [];
      const x = Math.max(0, box.originX / width); const y = Math.max(0, box.originY / height);
      return [{ x, y, width: Math.min(1, (box.originX + box.width) / width) - x, height: Math.min(1, (box.originY + box.height) / height) - y,
        score: detection.categories[0]?.score ?? 0 }];
    });
    self.postMessage({ type: 'result', id: message.id, observedAtMs: message.observedAtMs, boxes });
  } catch {
    self.postMessage({ type: 'error', id: message.id, message: 'The local face detector could not process this input. Restart tracking or use the pointer preview.' });
  } finally { bitmap?.close(); }
};
