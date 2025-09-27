import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { MessageCircle, X, Send, Bot, User, AlertTriangle } from "lucide-react";
import { FallAlert } from "@/services/api";

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  type?: 'normal' | 'alert' | 'action';
  alertId?: string;
}

interface ChatbotProps {
  alerts: FallAlert[];
  onResolveAlert: (alertId: string) => void;
  onTriggerEmergency: () => void;
}

const Chatbot = ({ alerts, onResolveAlert, onTriggerEmergency }: ChatbotProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: 'Hello! I\'m your AI Fall Detection System assistant. I can help you monitor alerts, understand system status, and provide emergency assistance. How can I help you today?',
      sender: 'bot',
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Add alert notifications when new alerts come in
  useEffect(() => {
    const latestAlert = alerts[0];
    if (latestAlert && !latestAlert.resolved) {
      const alertTime = new Date(latestAlert.timestamp).getTime();
      const now = Date.now();
      
      // Only show notification for alerts from the last 10 seconds
      if (now - alertTime < 10000) {
        const alertMessage: Message = {
          id: `alert-${latestAlert.id}`,
          text: `🚨 FALL ALERT: ${latestAlert.state} detected with ${(latestAlert.confidence * 100).toFixed(1)}% confidence at ${new Date(latestAlert.timestamp).toLocaleTimeString()}. Would you like me to help resolve this alert?`,
          sender: 'bot',
          timestamp: new Date(),
          type: 'alert',
          alertId: latestAlert.id
        };
        
        setMessages(prev => [...prev, alertMessage]);
      }
    }
  }, [alerts]);

  const handleSendMessage = async (text: string) => {
    if (!text.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: text.trim(),
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsTyping(true);

    // Simulate bot response delay
    setTimeout(() => {
      const botResponse = getBotResponse(text.trim().toLowerCase());
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: botResponse,
        sender: 'bot',
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, botMessage]);
      setIsTyping(false);
    }, 1000 + Math.random() * 1000);
  };

  const getBotResponse = (userInput: string): string => {
    const input = userInput.toLowerCase();
    
    // Emergency keywords
    if (input.includes('emergency') || input.includes('help') || input.includes('urgent')) {
      return 'I understand this may be urgent. I can:\n• Trigger an emergency alert immediately\n• Show you current unresolved alerts\n• Help you contact emergency services\n\nWould you like me to trigger an emergency alert now?';
    }
    
    // Alert-related queries
    if (input.includes('alert') || input.includes('fall detected')) {
      const unresolvedAlerts = alerts.filter(alert => !alert.resolved);
      if (unresolvedAlerts.length > 0) {
        return `There are currently ${unresolvedAlerts.length} unresolved alerts:\n${unresolvedAlerts.map(alert => 
          `• ${alert.state} detected at ${new Date(alert.timestamp).toLocaleTimeString()} (${(alert.confidence * 100).toFixed(1)}% confidence)`
        ).join('\n')}\n\nWould you like me to help resolve any of these alerts?`;
      } else {
        return 'No unresolved alerts at the moment. The system is monitoring continuously for any fall incidents.';
      }
    }
    
    // Status queries
    if (input.includes('status') || input.includes('how is') || input.includes('system')) {
      return `Current system status:\n• Total alerts today: ${alerts.length}\n• Unresolved alerts: ${alerts.filter(a => !a.resolved).length}\n• System is actively monitoring for falls\n\nEverything appears to be functioning normally.`;
    }
    
    // Fall detection explanation
    if (input.includes('fall') || input.includes('detect') || input.includes('how does')) {
      return 'The AI Fall Detection System uses advanced computer vision and pose estimation to monitor for potential falls in real-time. It analyzes body posture, movement patterns, and fall indicators to trigger alerts when necessary.';
    }
    
    // Camera/video issues
    if (input.includes('camera') || input.includes('video') || input.includes('stream')) {
      return 'The system processes live video feed from connected cameras. If you\'re having issues:\n• Ensure camera is connected\n• Check that the backend service is running\n• Verify camera permissions\n• Try restarting the detection system';
    }
    
    // Resolve alerts
    if (input.includes('resolve') || input.includes('clear') || input.includes('dismiss')) {
      const unresolvedAlerts = alerts.filter(alert => !alert.resolved);
      if (unresolvedAlerts.length > 0) {
        return `I can help resolve alerts. There are ${unresolvedAlerts.length} unresolved alerts. Please specify which alert you'd like to resolve, or say "resolve all" to clear all alerts.`;
      } else {
        return 'There are no unresolved alerts to clear at the moment.';
      }
    }

    // Default helpful responses
    const defaultResponses = [
      'I can help you with fall detection alerts, system status, emergency procedures, and troubleshooting. What do you need assistance with?',
      'I\'m here to help monitor your safety. You can ask me about current alerts, system status, or emergency procedures.',
      'How can I assist you with the fall detection system today? I can check alerts, explain system features, or help with emergencies.',
    ];
    
    return defaultResponses[Math.floor(Math.random() * defaultResponses.length)];
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(inputValue);
    }
  };

  return (
    <>
      {/* Chat Toggle Button */}
      {!isOpen && (
        <Button
          onClick={() => setIsOpen(true)}
          className="fixed bottom-6 right-6 h-14 w-14 rounded-full shadow-lg bg-primary hover:bg-primary/90 text-primary-foreground z-50"
          size="icon"
        >
          <MessageCircle className="h-6 w-6" />
        </Button>
      )}

      {/* Chat Interface */}
      {isOpen && (
        <Card className="fixed bottom-6 right-6 w-96 h-[500px] shadow-2xl z-50 flex flex-col border-2 border-primary/20">
          {/* Chat Header */}
          <div className="flex items-center justify-between p-4 border-b border-border bg-primary text-primary-foreground rounded-t-lg">
            <div className="flex items-center space-x-3">
              <div className="flex-shrink-0">
                <Bot className="h-6 w-6" />
              </div>
              <div>
                <h3 className="font-semibold">AI Assistant</h3>
                <p className="text-xs opacity-90">Fall Detection Support</p>
              </div>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsOpen(false)}
              className="h-8 w-8 text-primary-foreground hover:bg-primary-foreground/20"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>

          {/* Messages Area */}
          <ScrollArea className="flex-1 p-4">
            <div className="space-y-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`flex items-start space-x-2 max-w-[80%] ${
                      message.sender === 'user' ? 'flex-row-reverse space-x-reverse' : ''
                    }`}
                  >
                    <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                      message.sender === 'user' 
                        ? 'bg-primary text-primary-foreground' 
                        : 'bg-muted text-muted-foreground'
                    }`}>
                      {message.sender === 'user' ? (
                        <User className="h-4 w-4" />
                      ) : (
                        <Bot className="h-4 w-4" />
                      )}
                    </div>
                    <div
                      className={`px-3 py-2 rounded-lg text-sm whitespace-pre-wrap ${
                        message.sender === 'user'
                          ? 'bg-primary text-primary-foreground'
                          : message.type === 'alert'
                          ? 'bg-red-50 text-red-900 border border-red-200'
                          : 'bg-muted text-foreground'
                      }`}
                    >
                      {message.type === 'alert' && (
                        <div className="flex items-center mb-2">
                          <AlertTriangle className="w-4 h-4 mr-2 text-red-600" />
                          <span className="font-semibold text-red-600">FALL ALERT</span>
                        </div>
                      )}
                      {message.text}
                      
                      {/* Action buttons for alert messages */}
                      {message.type === 'alert' && message.alertId && (
                        <div className="flex space-x-2 mt-3">
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={(e) => {
                              e.preventDefault();
                              onResolveAlert(message.alertId!);
                              setMessages(prev => prev.map(msg => 
                                msg.id === message.id 
                                  ? { ...msg, text: msg.text + '\n\n✅ Alert resolved.' }
                                  : msg
                              ));
                            }}
                            className="text-xs"
                          >
                            Resolve Alert
                          </Button>
                          <Button
                            size="sm"
                            variant="destructive"
                            onClick={(e) => {
                              e.preventDefault();
                              onTriggerEmergency();
                              setMessages(prev => [...prev, {
                                id: Date.now().toString(),
                                text: '🚨 Emergency alert triggered! Emergency services have been notified.',
                                sender: 'bot',
                                timestamp: new Date(),
                                type: 'alert'
                              }]);
                            }}
                            className="text-xs"
                          >
                            Emergency
                          </Button>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
              
              {/* Typing Indicator */}
              {isTyping && (
                <div className="flex justify-start">
                  <div className="flex items-start space-x-2 max-w-[80%]">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-muted text-muted-foreground flex items-center justify-center">
                      <Bot className="h-4 w-4" />
                    </div>
                    <div className="px-3 py-2 rounded-lg bg-muted text-foreground">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                        <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>
          </ScrollArea>

          {/* Input Area */}
          <div className="p-4 border-t border-border">
            <div className="flex space-x-2">
              <Input
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask about the fall detection system..."
                className="flex-1"
                disabled={isTyping}
              />
              <Button
                onClick={() => handleSendMessage(inputValue)}
                disabled={!inputValue.trim() || isTyping}
                size="icon"
                className="bg-primary hover:bg-primary/90"
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              Ask me about system status, alerts, or troubleshooting
            </p>
          </div>
        </Card>
      )}
    </>
  );
};

export default Chatbot;