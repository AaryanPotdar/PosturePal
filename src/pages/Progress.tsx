
import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { supabase } from "@/integrations/supabase/client";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { PostureSession } from "@/types/database";

interface SessionData extends PostureSession {
  average_score: number;
  measurement_count: number;
}

const Progress = () => {
  const [sessions, setSessions] = useState<SessionData[]>([]);

  useEffect(() => {
    const fetchProgress = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) return;

      const { data: sessionsData, error } = await supabase
        .from('posture_sessions')
        .select(`
          *,
          posture_measurements(posture_score)
        `)
        .eq('user_id', user.id)
        .order('created_at', { ascending: false })
        .limit(10);

      if (error) {
        console.error('Error fetching progress:', error);
        return;
      }

      if (!sessionsData) return;

      const processedSessions = sessionsData.map(session => {
        const measurements = session.posture_measurements as { posture_score: number }[];
        const average = measurements.length > 0
          ? Math.round(measurements.reduce((sum, m) => sum + m.posture_score, 0) / measurements.length)
          : 0;

        return {
          ...session,
          average_score: average,
          measurement_count: measurements.length
        };
      });

      setSessions(processedSessions);
    };

    fetchProgress();
  }, []);

  return (
    <div className="max-w-screen-xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-neutral-900 mb-6">Your Progress</h1>
      
      <div className="space-y-6">
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4">Progress Over Time</h2>
          <div className="h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={sessions}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="created_at" 
                  tickFormatter={(date) => new Date(date).toLocaleDateString()}
                />
                <YAxis domain={[0, 100]} />
                <Tooltip 
                  labelFormatter={(label) => new Date(label).toLocaleString()}
                  formatter={(value: number) => [`${value}`, 'Average Score']}
                />
                <Line 
                  type="monotone" 
                  dataKey="average_score" 
                  stroke="#6366f1" 
                  name="Average Score"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <div className="grid gap-6 md:grid-cols-2">
          {sessions.map((session) => (
            <Card key={session.id} className="p-4">
              <div className="flex justify-between items-start mb-2">
                <div>
                  <h3 className="font-semibold">
                    Session on {new Date(session.created_at).toLocaleDateString()}
                  </h3>
                  <p className="text-sm text-neutral-500">
                    {new Date(session.created_at).toLocaleTimeString()}
                  </p>
                </div>
                <div className="text-right">
                  <p className="font-medium">Average Score</p>
                  <p className="text-2xl font-bold text-primary">
                    {session.average_score}
                  </p>
                </div>
              </div>
              <p className="text-sm text-neutral-600">
                {session.measurement_count} measurements taken
              </p>
              <p className="text-sm text-neutral-600">
                Duration: {session.ended_at 
                  ? Math.round((new Date(session.ended_at).getTime() - new Date(session.created_at).getTime()) / 60000)
                  : 'Ongoing'} minutes
              </p>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Progress;
