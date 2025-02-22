export interface PostureSession {
  id: string;
  user_id: string;
  created_at: string;
  ended_at: string | null;
  is_active: boolean;
}

export interface PostureMeasurement {
  id: string;
  session_id: string;
  user_id: string;
  posture_score: number;
  head_position: {
    x: number;
    y: number;
    score: number | null;
  } | null;
  shoulder_position: {
    left: {
      x: number;
      y: number;
      score: number | null;
    } | null;
    right: {
      x: number;
      y: number;
      score: number | null;
    } | null;
  } | null;
  spine_alignment: {
    top: {
      x: number;
      y: number;
      score: number | null;
    } | null;
    bottom: {
      x: number;
      y: number;
      score: number | null;
    } | null;
  } | null;
  head_tilt_detected: boolean;
  shoulders_uneven: boolean;
  head_too_low: boolean;
  head_too_forward: boolean;
  neck_tilt_angle: number;
  shoulder_angles: {
    left: number;
    right: number;
  };
  created_at: string;
}

export interface PostureSessionInsert {
  user_id: string;
  ended_at?: string | null;
  is_active?: boolean;
}

export interface PostureMeasurementInsert {
  session_id: string;
  user_id: string;
  posture_score: number;
  head_position?: {
    x: number;
    y: number;
    score: number | null;
  } | null;
  shoulder_position?: {
    left: {
      x: number;
      y: number;
      score: number | null;
    } | null;
    right: {
      x: number;
      y: number;
      score: number | null;
    } | null;
  } | null;
  spine_alignment?: {
    top: {
      x: number;
      y: number;
      score: number | null;
    } | null;
    bottom: {
      x: number;
      y: number;
      score: number | null;
    } | null;
  } | null;
  head_tilt_detected?: boolean;
  shoulders_uneven?: boolean;
  head_too_low?: boolean;
  head_too_forward?: boolean;
  neck_tilt_angle?: number;
  shoulder_angles?: {
    left: number;
    right: number;
  };
}

export interface PositionData {
  head: {
    x: number;
    y: number;
    score: number | null;
  } | null;
  shoulders: {
    left: {
      x: number;
      y: number;
      score: number | null;
    } | null;
    right: {
      x: number;
      y: number;
      score: number | null;
    } | null;
  } | null;
  spine: {
    top: {
      x: number;
      y: number;
      score: number | null;
    } | null;
    bottom: {
      x: number;
      y: number;
      score: number | null;
    } | null;
  } | null;
}
