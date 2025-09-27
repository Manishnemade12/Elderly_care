import Header from "@/components/Header";
import Footer from "@/components/Footer";
import LiveFeed from "@/components/LiveFeed";
import StatusPanel from "@/components/StatusPanel";
import EventLog from "@/components/EventLog";
import Chatbot from "@/components/Chatbot";
import { useFallDetection } from "@/hooks/useFallDetection";

const Index = () => {
  const fallDetection = useFallDetection();

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Header />
      
      <main className="flex-1 container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-10 gap-8 h-full min-h-[600px]">
          {/* Left Column - Video Feed (70% width) */}
          <div className="lg:col-span-7">
            <LiveFeed 
              isDetectionActive={fallDetection.isDetectionActive}
              systemStatus={fallDetection.systemStatus}
            />
          </div>
          
          {/* Right Column - Status and Event Log (30% width) */}
          <div className="lg:col-span-3 space-y-6">
            <StatusPanel 
              systemStatus={fallDetection.systemStatus}
              isDetectionActive={fallDetection.isDetectionActive}
              loading={fallDetection.loading}
              error={fallDetection.error}
              onStartDetection={fallDetection.startDetection}
              onStopDetection={fallDetection.stopDetection}
              onTriggerEmergency={fallDetection.triggerEmergency}
            />
            <EventLog events={fallDetection.events} />
          </div>
        </div>
      </main>
      
      <Footer />
      <Chatbot 
        alerts={fallDetection.alerts}
        onResolveAlert={fallDetection.resolveAlert}
        onTriggerEmergency={fallDetection.triggerEmergency}
      />
    </div>
  );
};

export default Index;
