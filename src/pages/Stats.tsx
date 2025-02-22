
import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { supabase } from "@/integrations/supabase/client";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface StatsData {
  date: string;
  averageScore: number;
  sessionCount: number;
}

const Stats = () => {
  const [statsData, setStatsData] = useState<StatsData[]>([]);

  useEffect(() => {
    const fetchStats = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) return;

      const { data: measurements, error } = await supabase
        .from('posture_measurements')
        .select('created_at, posture_score')
        .eq('user_id', user.id)
        .order('created_at');

      if (error) {
        console.error('Error fetching stats:', error);
        return;
      }

      if (!measurements) return;

      // Group measurements by date and calculate averages
      const groupedData = measurements.reduce((acc: { [key: string]: number[] }, curr) => {
        const date = new Date(curr.created_at).toLocaleDateString();
        if (!acc[date]) acc[date] = [];
        acc[date].push(curr.posture_score);
        return acc;
      }, {});

      const stats: StatsData[] = Object.entries(groupedData).map(([date, scores]) => ({
        date,
        averageScore: Math.round(scores.reduce((sum, score) => sum + score, 0) / scores.length),
        sessionCount: scores.length,
      }));

      setStatsData(stats);
    };

    fetchStats();
  }, []);

  return (
    <div className="max-w-screen-xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-neutral-900 mb-6">Statistics</h1>
      
      <div className="grid gap-6 md:grid-cols-2">
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4">Posture Score Over Time</h2>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={statsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Bar dataKey="averageScore" fill="#6366f1" name="Average Score" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4">Session Count</h2>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={statsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="sessionCount" fill="#8b5cf6" name="Number of Sessions" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default Stats;
