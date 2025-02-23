import { Outlet, useNavigate, useLocation } from "react-router-dom";
import { Home, Activity, PlayCircle, Settings, BarChart } from "lucide-react";

const Layout = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const navigation = [
    { name: "Home", path: "/", icon: Home },
    { name: "Analysis", path: "/analysis", icon: PlayCircle },
    { name: "Progress", path: "/progress", icon: Activity },
    { name: "Stats", path: "/stats", icon: BarChart },
    { name: "Settings", path: "/settings", icon: Settings },
  ];

  return (
    <div className="min-h-screen bg-neutral-100">
      <header className="bg-white border-b border-neutral-200">
        <div className="max-w-screen-xl mx-auto px-4">
          <div className="h-16 flex items-center">
            <img 
              src="/pplogo.png" 
              alt="PosturePal Logo" 
              className="h-8 w-auto cursor-pointer"
              onClick={() => navigate('/')}
            />
          </div>
        </div>
      </header>
      <main className="pb-20">
        <Outlet />
      </main>
      <nav className="fixed bottom-0 left-0 right-0 bg-white border-t border-neutral-200">
        <div className="max-w-screen-xl mx-auto px-4">
          <div className="flex justify-around items-center h-16">
            {navigation.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              return (
                <button
                  key={item.name}
                  onClick={() => navigate(item.path)}
                  className={`flex flex-col items-center justify-center w-16 h-16 transition-colors ${
                    isActive ? "text-primary" : "text-neutral-400 hover:text-neutral-600"
                  }`}
                >
                  <Icon className="w-6 h-6" />
                  <span className="text-xs mt-1">{item.name}</span>
                </button>
              );
            })}
          </div>
        </div>
      </nav>
    </div>
  );
};

export default Layout;
