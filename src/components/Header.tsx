import shieldIcon from "@/assets/shield-heart-icon.png";

const Header = () => {
  return (
    <header className="bg-gradient-to-r from-primary to-primary/90 shadow-sm border-b border-border">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center space-x-3">
          <img 
            src={shieldIcon} 
            alt="AI Fall Detection Shield Icon" 
            className="w-8 h-8 object-contain"
          />
          <h1 className="text-2xl font-bold text-primary-foreground">
            AI Fall Detection System
          </h1>
        </div>
      </div>
    </header>
  );
};

export default Header;