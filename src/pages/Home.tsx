import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowRight, Award, PlayCircle, Zap } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";

const Home = () => {
  const navigate = useNavigate();
  const [score, setScore] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchLatestScore = async () => {
      try {
        const { data: { user } } = await supabase.auth.getUser();
        if (!user) return;

        const { data: latestSession, error } = await supabase
          .from('posture_sessions')
          .select(`
            *,
            posture_measurements(posture_score)
          `)
          .eq('user_id', user.id)
          .order('created_at', { ascending: false })
          .limit(1)
          .single();

        if (error) {
          console.error('Error fetching latest session:', error);
          return;
        }

        if (latestSession) {
          const measurements = latestSession.posture_measurements as { posture_score: number }[];
          const average = measurements.length > 0
            ? Math.round(measurements.reduce((sum, m) => sum + m.posture_score, 0) / measurements.length)
            : null;
          setScore(average);
        }
      } catch (error) {
        console.error('Error:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchLatestScore();
  }, []);

  return (
    <div className="max-w-screen-xl mx-auto px-4 py-8 animate-fade-in">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-neutral-900">PosturePal</h1>
        <p className="text-neutral-600 mt-2">Your personal posture assistant</p>
      </div>

      <div className="bg-white rounded-2xl p-6 shadow-lg mb-8">
        <div className="flex items-center justify-between mb-4">
          <span className="text-sm font-medium text-neutral-500">Today's Posture Score</span>
          <Award className="text-primary w-6 h-6" />
        </div>
        {isLoading ? (
          <div className="h-16 flex items-center justify-center">
            <span className="text-neutral-600">Loading...</span>
          </div>
        ) : score !== null ? (
          <>
            <div className="flex items-baseline">
              <span className="text-4xl font-bold text-neutral-900">{score}</span>
              <span className="text-neutral-500 ml-2">/100</span>
            </div>
            <div className="mt-4 bg-neutral-100 rounded-full h-2">
              <div
                className="bg-primary rounded-full h-2 transition-all duration-500"
                style={{ width: `${score}%` }}
              />
            </div>
          </>
        ) : (
          <div className="text-center py-4">
            <p className="text-neutral-600 mb-2">No posture data available</p>
            <p className="text-sm text-neutral-500">Start an analysis session to see your score</p>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
        <button
          onClick={() => navigate("/analysis")}
          className="bg-primary text-white p-6 rounded-2xl flex items-center justify-between hover:bg-primary-dark transition-colors"
        >
          <div className="flex items-center">
            <PlayCircle className="w-6 h-6 mr-3" />
            <div className="text-left">
              <h3 className="font-semibold">Start Analysis</h3>
              <p className="text-sm opacity-90">Check your posture now</p>
            </div>
          </div>
          <ArrowRight className="w-5 h-5" />
        </button>

        <button
          onClick={() => navigate("/progress")}
          className="bg-white p-6 rounded-2xl flex items-center justify-between hover:bg-neutral-50 transition-colors border border-neutral-200"
        >
          <div className="flex items-center">
            <Zap className="w-6 h-6 mr-3 text-warning" />
            <div className="text-left">
              <h3 className="font-semibold text-neutral-900">View Progress</h3>
              <p className="text-sm text-neutral-600">Track your improvement</p>
            </div>
          </div>
          <ArrowRight className="w-5 h-5 text-neutral-400" />
        </button>
      </div>

      <div className="bg-white rounded-2xl p-6 border border-neutral-200">
        <h3 className="font-semibold text-neutral-900 mb-4">Daily Tip</h3>
        <p className="text-neutral-600">
          Remember to take regular breaks and stretch every hour to maintain good posture throughout
          your day.
        </p>
      </div>
    </div>
  );
};

export default Home;
