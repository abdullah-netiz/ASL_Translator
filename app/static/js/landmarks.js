/**
 * MediaPipe Holistic landmark extraction & normalization in JavaScript.
 * MUST maintain bit-exact algorithmic parity with Python `src/extract/normalize.py`.
 *
 * Feature vector layout (1200 floats total):
 *   Static block (600 floats):
 *     - Pose (17 points * 4): 68 floats (x, y, z, visibility), shoulder-centered and scaled
 *     - Left hand (21 points * 4): 84 floats, shoulder-centered and scaled
 *     - Right hand (21 points * 4): 84 floats, shoulder-centered and scaled
 *     - Face subset (59 points * 4): 236 floats, shoulder-centered and scaled
 *     - Left hand wrist-relative (21 points * 3): 63 floats (x, y, z relative to wrist)
 *     - Right hand wrist-relative (21 points * 3): 63 floats (x, y, z relative to wrist)
 *     - Hand presence flags: 2 floats [left_detected, right_detected] (0.0 or 1.0)
 *   Velocity block (600 floats):
 *     - delta(static[t] - static[t-1]), zero at t=0
 */

// Selected landmark indices matching configs/landmarks.yaml
export const POSE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
export const LEFT_SHOULDER_IDX = 11;
export const RIGHT_SHOULDER_IDX = 12;

export const FACE_INDICES = [
  // Lips outer (20 pts)
  61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
  // Eyebrows (10 pts)
  70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
  // Left eye (6 pts)
  33, 246, 161, 160, 159, 158,
  // Right eye (6 pts)
  362, 398, 384, 385, 386, 387,
  // Face contour (17 pts)
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400
];

export const STATIC_DIM = 600;
export const TOTAL_DIM = 1200;

export class LandmarkNormalizer {
  constructor() {
    this.prevStatic = null;
  }

  reset() {
    this.prevStatic = null;
  }

  /**
   * Process Holistic detection results for a single frame.
   * @param {Object} results - MediaPipe Holistic results object
   * @returns {Float32Array} 1200-element feature vector
   */
  processFrame(results) {
    const poseLandmarks = results.poseLandmarks || [];
    const leftHandLandmarks = results.leftHandLandmarks || [];
    const rightHandLandmarks = results.rightHandLandmarks || [];
    const faceLandmarks = results.faceLandmarks || [];

    const hasPose = poseLandmarks.length >= 13;
    const hasLh = leftHandLandmarks.length === 21;
    const hasRh = rightHandLandmarks.length === 21;

    // 1. Calculate mid-shoulder and shoulder scale
    let midX = 0, midY = 0, midZ = 0, scale = 1.0;
    if (hasPose) {
      const ls = poseLandmarks[LEFT_SHOULDER_IDX];
      const rs = poseLandmarks[RIGHT_SHOULDER_IDX];
      midX = (ls.x + rs.x) / 2.0;
      midY = (ls.y + rs.y) / 2.0;
      midZ = (ls.z + rs.z) / 2.0;
      const dx = ls.x - rs.x;
      const dy = ls.y - rs.y;
      const dz = ls.z - rs.z;
      scale = Math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6;
    }

    const staticVec = new Float32Array(STATIC_DIM);
    let offset = 0;

    // Helper to normalize and push landmarks
    const pushNormalized = (landmarks, indices, numPts) => {
      for (let i = 0; i < numPts; i++) {
        const idx = indices ? indices[i] : i;
        const lm = (landmarks && landmarks[idx]) ? landmarks[idx] : null;
        if (lm) {
          staticVec[offset++] = (lm.x - midX) / scale;
          staticVec[offset++] = (lm.y - midY) / scale;
          staticVec[offset++] = (lm.z - midZ) / scale;
          staticVec[offset++] = lm.visibility !== undefined ? lm.visibility : 1.0;
        } else {
          staticVec[offset++] = 0.0;
          staticVec[offset++] = 0.0;
          staticVec[offset++] = 0.0;
          staticVec[offset++] = 0.0;
        }
      }
    };

    // 2. Pose subset (17 points * 4 = 68)
    pushNormalized(poseLandmarks, POSE_INDICES, POSE_INDICES.length);

    // 3. Hands shoulder-normalized (21 * 4 = 84 each)
    const lhStart = offset;
    pushNormalized(leftHandLandmarks, null, 21);
    const rhStart = offset;
    pushNormalized(rightHandLandmarks, null, 21);

    // 4. Face subset (59 points * 4 = 236)
    pushNormalized(faceLandmarks, FACE_INDICES, FACE_INDICES.length);

    // 5. Wrist-relative hands (21 * 3 = 63 each)
    // Landmark 0 of each hand is the wrist in shoulder-normalized coords
    const lhWristX = hasLh ? staticVec[lhStart] : 0.0;
    const lhWristY = hasLh ? staticVec[lhStart + 1] : 0.0;
    const lhWristZ = hasLh ? staticVec[lhStart + 2] : 0.0;

    for (let i = 0; i < 21; i++) {
      if (hasLh) {
        staticVec[offset++] = staticVec[lhStart + i * 4] - lhWristX;
        staticVec[offset++] = staticVec[lhStart + i * 4 + 1] - lhWristY;
        staticVec[offset++] = staticVec[lhStart + i * 4 + 2] - lhWristZ;
      } else {
        staticVec[offset++] = 0.0;
        staticVec[offset++] = 0.0;
        staticVec[offset++] = 0.0;
      }
    }

    const rhWristX = hasRh ? staticVec[rhStart] : 0.0;
    const rhWristY = hasRh ? staticVec[rhStart + 1] : 0.0;
    const rhWristZ = hasRh ? staticVec[rhStart + 2] : 0.0;

    for (let i = 0; i < 21; i++) {
      if (hasRh) {
        staticVec[offset++] = staticVec[rhStart + i * 4] - rhWristX;
        staticVec[offset++] = staticVec[rhStart + i * 4 + 1] - rhWristY;
        staticVec[offset++] = staticVec[rhStart + i * 4 + 2] - rhWristZ;
      } else {
        staticVec[offset++] = 0.0;
        staticVec[offset++] = 0.0;
        staticVec[offset++] = 0.0;
      }
    }

    // 6. Presence masks (2 floats)
    staticVec[offset++] = hasLh ? 1.0 : 0.0;
    staticVec[offset++] = hasRh ? 1.0 : 0.0;

    // 7. Calculate velocity (delta from previous static vector)
    const outVec = new Float32Array(TOTAL_DIM);
    outVec.set(staticVec, 0);

    if (this.prevStatic !== null) {
      for (let i = 0; i < STATIC_DIM; i++) {
        outVec[STATIC_DIM + i] = staticVec[i] - this.prevStatic[i];
      }
    } else {
      // First frame: zero velocity
      outVec.fill(0.0, STATIC_DIM, TOTAL_DIM);
    }

    this.prevStatic = new Float32Array(staticVec);
    return outVec;
  }
}
