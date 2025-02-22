
import { Button } from "@/components/ui/button";
import { supabase } from "@/integrations/supabase/client";
import { useNavigate } from "react-router-dom";

const Settings = () => {
  const navigate = useNavigate();

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    navigate("/auth");
  };

  return (
    <div className="max-w-screen-xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-neutral-900 mb-4">Settings</h1>
      <div className="space-y-6">
        <div className="space-y-2">
          <h2 className="text-xl font-semibold text-neutral-800">Account</h2>
          <Button 
            variant="destructive"
            onClick={handleSignOut}
          >
            Sign Out
          </Button>
        </div>
      </div>
    </div>
  );
};

export default Settings;
