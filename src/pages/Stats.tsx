import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { supabase } from "@/integrations/supabase/client";
import { Loader2 } from "lucide-react";
import { PostureMeasurementInsert } from "@/types/database";

interface SessionAnalytics {
  averageScore: number;
  issueCount: { [key: string]: number };
  totalMeasurements: number;
  sessionDuration: string;
  aiFeedback: string;
}

const Stats = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [analytics, setAnalytics] = useState<SessionAnalytics | null>(null);
  const [error, setError] = useState<string | null>(null);

  const generateAnalytics = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        setError("Please sign in to view your analytics");
        return;
      }

      // Get the latest session
      const { data: sessions, error: sessionError } = await supabase
        .from('posture_sessions')
        .select('id, created_at, ended_at')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false })
        .limit(1);

      if (sessionError) throw sessionError;
      if (!sessions || sessions.length === 0) {
        setError("No sessions found. Please complete a posture analysis session first.");
        return;
      }

      const lastSession = sessions[0];
      
      // Get all measurements for the last session
      const { data: measurements, error: measurementError } = await supabase
        .from('posture_measurements')
        .select('posture_score, posture_issues, created_at')
        .eq('session_id', lastSession.id)
        .order('created_at')
        .returns<{ 
          posture_score: number; 
          posture_issues: string[] | null; 
          created_at: string; 
        }[]>();

      if (measurementError) throw measurementError;
      if (!measurements || measurements.length === 0) {
        setError("No measurements found for the last session.");
        return;
      }

      console.log('Raw measurements:', measurements); // Debug log

      // Calculate analytics
      const scores = measurements.map(m => m.posture_score);
      const averageScore = Math.round(
        scores.reduce((sum, score) => sum + score, 0) / scores.length
      );

      // Count issues - Updated handling
      const issueCount: { [key: string]: number } = {};
      console.log('Starting to process measurements for issues...');
      measurements.forEach((m, index) => {
        console.log(`Processing measurement ${index + 1}:`, {
          posture_issues: m.posture_issues,
          isArray: Array.isArray(m.posture_issues)
        });
        const issues = m.posture_issues || [];
        if (Array.isArray(issues)) {
          issues.forEach(issue => {
            if (issue) {
              issueCount[issue] = (issueCount[issue] || 0) + 1;
              console.log(`Added issue: ${issue}, count now: ${issueCount[issue]}`);
            }
          });
        }
      });

      console.log('Final issue count:', issueCount);

      // Calculate session duration
      const startTime = new Date(lastSession.created_at);
      const endTime = new Date(lastSession.ended_at || measurements[measurements.length - 1].created_at);
      const durationMinutes = Math.round((endTime.getTime() - startTime.getTime()) / (1000 * 60));

      // Prepare prompt for AI
      const prompt = `Analyze this posture session and provide specific feedback and exercises:
      Session Duration: ${durationMinutes} minutes
      Average Posture Score: ${averageScore}
      Detected Issues and Frequencies:
      ${Object.entries(issueCount)
        .map(([issue, count]) => `- ${issue}: ${count} times`)
        .join('\n')}
      
      Please provide:
      1. A brief analysis of the user's posture
      2. Specific exercises or stretches that target their problem areas
      3. Tips for maintaining better posture during daily activities
      4. Recommendations for improvement`;

      console.log('Sending prompt to Mistral:', prompt);

      // Get AI feedback
      try {
        console.log('Making request to Mistral API with prompt:', prompt);
        
        const requestBody = { prompt };
        console.log('Request body:', requestBody);

        const response = await fetch('http://127.0.0.1:5000/query_mistral', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify(requestBody)
        });

        console.log('API Response status:', response.status);
        console.log('API Response headers:', Object.fromEntries(response.headers.entries()));
        
        const responseText = await response.text();
        console.log('API Raw response text:', responseText);

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}, response: ${responseText}`);
        }

        let aiResponse;
        try {
          aiResponse = JSON.parse(responseText);
          console.log('Parsed AI response:', aiResponse);
        } catch (e) {
          console.error('Failed to parse JSON response:', e);
          throw new Error(`Invalid JSON response from API: ${responseText}`);
        }

        // Check the response structure
        if (!aiResponse || typeof aiResponse !== 'object') {
          throw new Error(`Invalid response format: ${JSON.stringify(aiResponse)}`);
        }

        // Get the text from the outputs array
        const completion = aiResponse.outputs?.[0]?.text || aiResponse.text;
                         
        if (!completion) {
          console.error('Full AI response:', aiResponse);
          throw new Error('No completion found in AI response. Response: ' + JSON.stringify(aiResponse));
        }

        setAnalytics({
          averageScore,
          issueCount,
          totalMeasurements: measurements.length,
          sessionDuration: `${durationMinutes} minutes`,
          aiFeedback: completion
        });

      } catch (err) {
        console.error('Detailed error getting AI feedback:', err);
        console.error('Error type:', err.constructor.name);
        console.error('Error message:', err.message);
        if (err instanceof Error && err.stack) {
          console.error('Stack trace:', err.stack);
        }
        
        // Still show analytics but with error message for AI feedback
        setAnalytics({
          averageScore,
          issueCount,
          totalMeasurements: measurements.length,
          sessionDuration: `${durationMinutes} minutes`,
          aiFeedback: `Failed to get AI analysis: ${err.message}`
        });
      }

    } catch (err) {
      console.error('Error generating analytics:', err);
      setError(err instanceof Error ? err.message : 'An error occurred while generating analytics');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-screen-xl mx-auto px-4 py-8">
      <div className="flex flex-col items-center space-y-6">
        <h1 className="text-3xl font-bold text-neutral-900">Session Analytics</h1>
        
        <Button 
          onClick={generateAnalytics} 
          disabled={isLoading}
          className="w-full max-w-md"
        >
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Analyzing Session...
            </>
          ) : (
            'Generate Session Analytics'
          )}
        </Button>

        {error && (
          <Card className="w-full max-w-3xl p-6 bg-red-50">
            <p className="text-red-600">{error}</p>
          </Card>
        )}

        {analytics && !error && (
          <Card className="w-full max-w-3xl p-6 space-y-6">
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <h2 className="text-xl font-semibold">Last Session Overview</h2>
                <span className="text-gray-500">Duration: {analytics.sessionDuration}</span>
              </div>
              
              <div className="flex justify-between items-center p-4 bg-gray-50 rounded-lg">
                <span className="text-lg">Average Posture Score</span>
                <span 
                  className={`text-3xl font-bold ${
                    analytics.averageScore >= 90 ? 'text-green-500' :
                    analytics.averageScore >= 80 ? 'text-yellow-500' :
                    analytics.averageScore >= 70 ? 'text-orange-500' :
                    'text-red-500'
                  }`}
                >
                  {analytics.averageScore}
                </span>
              </div>

              <div className="space-y-2">
                <h3 className="text-lg font-semibold">Detected Issues</h3>
                {Object.keys(analytics.issueCount).length === 0 ? (
                  <p className="text-gray-500 italic">No posture issues detected in this session.</p>
                ) : (
                  <>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {Object.entries(analytics.issueCount).map(([issue, count]) => (
                        <div key={issue} className="flex justify-between p-4 bg-gray-50 rounded-lg border border-gray-200">
                          <span className="font-medium text-gray-700">{issue}</span>
                          <span className="font-semibold text-red-500">{count} times</span>
                        </div>
                      ))}
                    </div>
                    <p className="text-sm text-gray-500 mt-2">
                      Total unique issues detected: {Object.keys(analytics.issueCount).length}
                    </p>
                  </>
                )}
              </div>

              <div className="space-y-2">
                <h3 className="text-lg font-semibold">AI Analysis & Recommendations</h3>
                <div className="prose max-w-none">
                  <p className="whitespace-pre-wrap">{analytics.aiFeedback}</p>
                </div>
              </div>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
};

export default Stats;
