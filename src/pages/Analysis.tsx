import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { toast } from "@/components/ui/use-toast";
import { supabase } from "@/integrations/supabase/client";
import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection';
import { PostureSessionInsert, PostureMeasurementInsert, PositionData } from "@/types/database";

// Add Math.degrees type declaration
declare global {
  interface Math {
    degrees(radians: number): number;
  }
}

const MEASUREMENT_THRESHOLD = {
  SHOULDER_ANGLE: 5,  // degrees
  NECK_ANGLE: 5,      // degrees
  HEAD_POSITION: 10,  // pixels
  SHOULDER_HEIGHT: 10 // pixels
};

type PostureMetrics = {
  shoulderAngle: number;
  neckAngle: number;
  headPosition: number;
  shoulderEarDist: number;
};

const Analysis = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [detector, setDetector] = useState<poseDetection.PoseDetector | null>(null);
  const [currentScore, setCurrentScore] = useState<number | null>(null);
  const [postureIssues, setPostureIssues] = useState<string[]>([]);
  const lastMeasurementTime = useRef<number | null>(null);
  const lastNotificationTime = useRef<number | null>(null);
  const notificationSound = useRef<HTMLAudioElement | null>(null);
  const lastMetrics = useRef<PostureMetrics | null>(null);
  
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
      setIsAnalyzing(false);

      const { error } = await supabase
        .from('posture_sessions')
        .update({ 
          ended_at: new Date().toISOString(), 
          is_active: false 
        })
        .eq('id', sessionId);

      if (error) throw error;
      
      setSessionId(null);
      setCurrentScore(null);
      lastMeasurementTime.current = null;

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

  const saveMeasurement = async (score: number, positions: PositionData, issues: string[]) => {
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
        posture_issues: issues
      };

      const { error } = await supabase
        .from('posture_measurements')
        .insert(measurement);

      if (error) throw error;
      
      // Update current score
      setCurrentScore(score);
    } catch (error) {
      console.error('Error saving measurement:', error);
    }
  };

  useEffect(() => {
    const setupCamera = async () => {
      if (!videoRef.current || !canvasRef.current) return;

      try {
        await tf.ready();
        
        const detectorConfig = {
          modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING
        };
        const detector = await poseDetection.createDetector(
          poseDetection.SupportedModels.MoveNet,
          detectorConfig
        );
        setDetector(detector);

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
    if (!detector || !videoRef.current || !canvasRef.current) return;

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

      // Only run pose detection if we're analyzing
      if (isAnalyzing) {
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

          // Extract positions first
          const positions = extractPositions(pose.keypoints);
          
          // Calculate posture score and get issues
          const result = await calculatePostureScore(pose.keypoints);
          if (result) {
            setCurrentScore(result.posture_score);
            const currentIssues = result.posture_issues || [];
            setPostureIssues(currentIssues);

            // Save to database every second
            const now = Date.now();
            if (!lastMeasurementTime.current || now - lastMeasurementTime.current >= 1000) {
              await saveMeasurement(result.posture_score, positions, currentIssues);
              lastMeasurementTime.current = now;
            }

            // Check for low score and show notification
            if (result.posture_score < 60 && (!lastNotificationTime.current || now - lastNotificationTime.current >= 10000)) {
              notificationSound.current?.play().catch(err => console.error('Error playing sound:', err));
              
              toast({
                variant: "destructive",
                title: "⚠️ Poor Posture Alert!",
                description: (
                  <div className="space-y-2">
                    <p className="font-bold text-lg">Score: {result.posture_score}</p>
                    <p>Please adjust your position:</p>
                    <ul className="list-disc pl-4">
                      {currentIssues.map((issue, index) => (
                        <li key={index}>{issue}</li>
                      ))}
                    </ul>
                  </div>
                ),
                duration: 5000,
              });
              lastNotificationTime.current = now;
            }
          }
        }
      }
      
      // Always continue the animation loop
      animationFrame = requestAnimationFrame(detectPose);
    };

    detectPose();

    return () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame);
      }
      lastMeasurementTime.current = null;
    };
  }, [detector, isAnalyzing]);

  const calculatePostureScore = async (keypoints: poseDetection.Keypoint[]) => {
    const nose = keypoints.find(kp => kp.name === 'nose');
    const leftShoulder = keypoints.find(kp => kp.name === 'left_shoulder');
    const rightShoulder = keypoints.find(kp => kp.name === 'right_shoulder');
    const leftEar = keypoints.find(kp => kp.name === 'left_ear');
    const rightEar = keypoints.find(kp => kp.name === 'right_ear');

    if (nose && leftShoulder && rightShoulder && leftEar && rightEar) {
      try {
        const videoWidth = videoRef.current?.videoWidth || 640;
        const videoHeight = videoRef.current?.videoHeight || 480;

        const postureData = {
          nose_x: nose.x,
          nose_y: nose.y,
          left_shoulder_x: leftShoulder.x,
          left_shoulder_y: leftShoulder.y,
          right_shoulder_x: rightShoulder.x,
          right_shoulder_y: rightShoulder.y,
          left_ear_x: leftEar.x,
          left_ear_y: leftEar.y,
          right_ear_x: rightEar.x,
          right_ear_y: rightEar.y,
          videoWidth,
          videoHeight
        };

        const response = await fetch('http://127.0.0.1:5000/predict_posture', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(postureData)
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
      } catch (error) {
        console.error('Error getting posture score from API:', error);
        return null;
      }
    }
    return null;
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

  // Initialize notification sound
  useEffect(() => {
    notificationSound.current = new Audio('/notif.mp3');
    notificationSound.current.volume = 0.5;
  }, []);

  return (
    <div className="max-w-screen-xl mx-auto px-4 py-8">
      <div className="flex flex-col items-center space-y-6">
        <h1 className="text-3xl font-bold text-neutral-900">Posture Analysis</h1>
        
        {currentScore !== null && isAnalyzing && (
          <Card className="w-full max-w-3xl p-6 mb-4">
            <div className="flex justify-between items-center">
              <h2 className="text-xl font-semibold">Current Posture Score</h2>
              <span 
                className={`text-4xl font-bold ${
                  currentScore >= 90 ? 'text-green-500' :
                  currentScore >= 80 ? 'text-yellow-500' :
                  currentScore >= 70 ? 'text-orange-500' :
                  'text-red-500'
                }`}
              >
                {currentScore}
              </span>
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