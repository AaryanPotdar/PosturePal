import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { toast } from "@/components/ui/use-toast";
import { supabase } from "@/integrations/supabase/client";
import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection';
import { PostureSessionInsert, PostureMeasurementInsert, PositionData } from "@/types/database";

const Analysis = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [detector, setDetector] = useState<poseDetection.PoseDetector | null>(null);
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [currentScore, setCurrentScore] = useState<number | null>(null);
  const [postureIssues, setPostureIssues] = useState<string[]>([]);
  
  const startSession = async () => {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) return;

      const { data, error } = await supabase
        .from('posture_sessions')
        .insert<PostureSessionInsert>({
          user_id: user.id,
          is_active: true
        })
        .select()
        .single();

      if (error) throw error;
      setSessionId(data?.id ?? null);
      setIsAnalyzing(true);
      toast({
        title: "Session Started",
        description: "Your posture is now being analyzed.",
      });
    } catch (error) {
      console.error('Error starting session:', error);
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to start analysis session.",
      });
    }
  };

  const stopSession = async () => {
    if (!sessionId) return;

    try {
      const { error } = await supabase
        .from('posture_sessions')
        .update({ 
          ended_at: new Date().toISOString(), 
          is_active: false 
        })
        .eq('id', sessionId);

      if (error) throw error;
      setIsAnalyzing(false);
      setSessionId(null);
      toast({
        title: "Session Ended",
        description: "Your posture analysis session has been saved.",
      });
    } catch (error) {
      console.error('Error ending session:', error);
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to end analysis session.",
      });
    }
  };

  const saveMeasurement = async (score: number, positions: PositionData, issues: any) => {
    if (!sessionId) return;

    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) return;

      const measurement: PostureMeasurementInsert = {
        session_id: sessionId,
        user_id: user.id,
        posture_score: score,
        head_position: positions.head,
        shoulder_position: positions.shoulders,
        spine_alignment: positions.spine,
        head_tilt_detected: issues.headTilt,
        shoulders_uneven: issues.shouldersUneven,
        head_too_low: issues.headTooLow,
        head_too_forward: issues.headTooForward,
        neck_tilt_angle: issues.neckTiltAngle,
        shoulder_angles: {
          left: issues.leftShoulderAngle,
          right: issues.rightShoulderAngle
        }
      };

      const { error } = await supabase
        .from('posture_measurements')
        .insert(measurement);

      if (error) throw error;
      
      setCurrentScore(score);
      setPostureIssues(Object.entries(issues)
        .filter(([key, value]) => value === true)
        .map(([key]) => key.replace(/([A-Z])/g, ' $1').trim()));
    } catch (error) {
      console.error('Error saving measurement:', error);
    }
  };

  useEffect(() => {
    const setupCamera = async () => {
      if (!videoRef.current || !canvasRef.current) return;

      try {
        await tf.ready();
        
        // Load both pose detector and our RL model
        const detectorConfig = {
          modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING
        };
        const detector = await poseDetection.createDetector(
          poseDetection.SupportedModels.MoveNet,
          detectorConfig
        );
        setDetector(detector);

        // Load our trained RL model
        const rlModel = await tf.loadLayersModel('posture_model.h5');
        setModel(rlModel);

        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { width: 1280, height: 720 } 
        });
        videoRef.current.srcObject = stream;
        videoRef.current.play();

      } catch (error) {
        console.error('Error setting up camera:', error);
        toast({
          variant: "destructive",
          title: "Camera Error",
          description: "Failed to access your camera. Please check permissions.",
        });
      }
    };

    setupCamera();

    return () => {
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    if (!detector || !model || !videoRef.current || !canvasRef.current) return;

    let animationFrame: number;

    const detectPose = async () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState !== 4) {
        animationFrame = requestAnimationFrame(detectPose);
        return;
      }

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const poses = await detector.estimatePoses(video);

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      if (poses.length > 0) {
        const pose = poses[0];
        
        // Draw keypoints
        pose.keypoints.forEach(keypoint => {
          if (keypoint.score && keypoint.score > 0.3) {
            ctx.beginPath();
            ctx.arc(keypoint.x, keypoint.y, 5, 0, 2 * Math.PI);
            ctx.fillStyle = 'red';
            ctx.fill();
          }
        });

        if (isAnalyzing) {
          const features = extractFeatures(pose.keypoints);
          const positions = extractPositions(pose.keypoints);
          
          // Get prediction from our RL model
          const prediction = model.predict(tf.tensor2d([features])) as tf.Tensor;
          const score = Math.round(prediction.dataSync()[0] * 100);
          
          // Calculate additional posture metrics
          const issues = analyzePosture(pose.keypoints);
          
          saveMeasurement(score, positions, issues);
        }
      }

      animationFrame = requestAnimationFrame(detectPose);
    };

    detectPose();

    return () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame);
      }
    };
  }, [detector, model, isAnalyzing]);

  const extractFeatures = (keypoints: poseDetection.Keypoint[]) => {
    const nose = keypoints.find(kp => kp.name === 'nose');
    const leftShoulder = keypoints.find(kp => kp.name === 'left_shoulder');
    const rightShoulder = keypoints.find(kp => kp.name === 'right_shoulder');
    const leftEar = keypoints.find(kp => kp.name === 'left_ear');
    const rightEar = keypoints.find(kp => kp.name === 'right_ear');
    
    if (!nose || !leftShoulder || !rightShoulder || !leftEar || !rightEar) {
      return [0, 0, 0, 0, 0, 0, 0];
    }

    const midShoulderX = (leftShoulder.x + rightShoulder.x) / 2;
    const midShoulderY = (leftShoulder.y + rightShoulder.y) / 2;
    
    const distNoseShoulders = distance2D(nose.x, nose.y, midShoulderX, midShoulderY);
    const shoulderWidth = distance2D(leftShoulder.x, leftShoulder.y, rightShoulder.x, rightShoulder.y);
    const ratio = distNoseShoulders / shoulderWidth;
    const neckTiltAngle = angleABC(leftEar.x, leftEar.y, nose.x, nose.y, rightEar.x, rightEar.y);
    const distLeftEarNose = distance2D(leftEar.x, leftEar.y, nose.x, nose.y);
    const distRightEarNose = distance2D(rightEar.x, rightEar.y, nose.x, nose.y);
    const angleLeftShoulder = angleABC(leftEar.x, leftEar.y, leftShoulder.x, leftShoulder.y, nose.x, nose.y);
    const angleRightShoulder = angleABC(rightEar.x, rightEar.y, rightShoulder.x, rightShoulder.y, nose.x, nose.y);

    return [
      distNoseShoulders,
      ratio,
      neckTiltAngle,
      distLeftEarNose,
      distRightEarNose,
      angleLeftShoulder,
      angleRightShoulder
    ];
  };

  const distance2D = (x1: number, y1: number, x2: number, y2: number) => {
    return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
  };

  const angleABC = (Ax: number, Ay: number, Bx: number, By: number, Cx: number, Cy: number) => {
    const ABx = Ax - Bx;
    const ABy = Ay - By;
    const CBx = Cx - Bx;
    const CBy = Cy - By;
    const dot = ABx * CBx + ABy * CBy;
    const magAB = Math.sqrt(ABx ** 2 + ABy ** 2);
    const magCB = Math.sqrt(CBx ** 2 + CBy ** 2);
    if (magAB === 0 || magCB === 0) return 180;
    const cosTheta = Math.min(Math.max(dot / (magAB * magCB), -1), 1);
    return (Math.acos(cosTheta) * 180) / Math.PI;
  };

  const analyzePosture = (keypoints: poseDetection.Keypoint[]) => {
    const nose = keypoints.find(kp => kp.name === 'nose');
    const leftShoulder = keypoints.find(kp => kp.name === 'left_shoulder');
    const rightShoulder = keypoints.find(kp => kp.name === 'right_shoulder');
    const leftEar = keypoints.find(kp => kp.name === 'left_ear');
    const rightEar = keypoints.find(kp => kp.name === 'right_ear');

    if (!nose || !leftShoulder || !rightShoulder || !leftEar || !rightEar) {
      return {
        headTilt: false,
        shouldersUneven: false,
        headTooLow: false,
        headTooForward: false,
        neckTiltAngle: 0,
        leftShoulderAngle: 0,
        rightShoulderAngle: 0
      };
    }

    const shoulderWidth = distance2D(leftShoulder.x, leftShoulder.y, rightShoulder.x, rightShoulder.y);
    const headHeight = distance2D(nose.x, nose.y, (leftShoulder.x + rightShoulder.x) / 2, (leftShoulder.y + rightShoulder.y) / 2);
    const earHeightDiff = Math.abs(leftEar.y - rightEar.y);
    const shoulderHeightDiff = Math.abs(leftShoulder.y - rightShoulder.y);

    const headTooLowThresh = shoulderWidth * 0.6;
    const headTooFarThresh = shoulderWidth * 1.2;
    const earLevelThresh = 15;
    const shoulderLevelThresh = 20;

    return {
      headTilt: earHeightDiff > earLevelThresh,
      shouldersUneven: shoulderHeightDiff > shoulderLevelThresh,
      headTooLow: headHeight < headTooLowThresh,
      headTooForward: headHeight > headTooFarThresh,
      neckTiltAngle: angleABC(leftEar.x, leftEar.y, nose.x, nose.y, rightEar.x, rightEar.y),
      leftShoulderAngle: angleABC(leftEar.x, leftEar.y, leftShoulder.x, leftShoulder.y, nose.x, nose.y),
      rightShoulderAngle: angleABC(rightEar.x, rightEar.y, rightShoulder.x, rightShoulder.y, nose.x, nose.y)
    };
  };

  const extractPositions = (keypoints: poseDetection.Keypoint[]): PositionData => {
    const getKeypoint = (name: string) => {
      const kp = keypoints.find(kp => kp.name === name);
      return kp ? { x: kp.x, y: kp.y, score: kp.score } : null;
    };

    return {
      head: getKeypoint('nose'),
      shoulders: {
        left: getKeypoint('left_shoulder'),
        right: getKeypoint('right_shoulder')
      },
      spine: {
        top: getKeypoint('shoulders'),
        bottom: getKeypoint('hips')
      }
    };
  };

  return (
    <div className="max-w-screen-xl mx-auto px-4 py-8">
      <div className="flex flex-col items-center space-y-6">
        <h1 className="text-3xl font-bold text-neutral-900">Posture Analysis</h1>
        
        {currentScore !== null && isAnalyzing && (
          <Card className="w-full max-w-3xl p-6 mb-4">
            <div className="flex flex-col space-y-4">
              <div className="flex justify-between items-center">
                <h2 className="text-xl font-semibold">Current Posture Score</h2>
                <span className="text-4xl font-bold text-primary">{currentScore}</span>
              </div>
              
              {postureIssues.length > 0 && (
                <div className="mt-4">
                  <h3 className="text-lg font-medium mb-2">Posture Issues Detected:</h3>
                  <ul className="list-disc list-inside space-y-1">
                    {postureIssues.map((issue, index) => (
                      <li key={index} className="text-red-600">{issue}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </Card>
        )}

        <Card className="w-full max-w-3xl p-4">
          <div className="relative aspect-video">
            <video
              ref={videoRef}
              className="absolute top-0 left-0 w-full h-full hidden"
              playsInline
            />
            <canvas
              ref={canvasRef}
              className="w-full h-full"
              width={1280}
              height={720}
            />
          </div>
        </Card>

        <div className="flex gap-4">
          {!isAnalyzing ? (
            <Button onClick={startSession}>
              Start Analysis
            </Button>
          ) : (
            <Button variant="destructive" onClick={stopSession}>
              Stop Analysis
            </Button>
          )}
        </div>
      </div>
    </div>
  );
};

export default Analysis;
