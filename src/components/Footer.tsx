const Footer = () => {
  return (
    <footer className="bg-muted/30 border-t border-border py-4">
      <div className="container mx-auto px-6">
        <div className="flex flex-col md:flex-row justify-between items-center space-y-2 md:space-y-0">
          <div className="text-sm text-muted-foreground">
            © 2024 AI Fall Detection System. Monitoring elderly safety with advanced AI technology.
          </div>
          <div className="flex items-center space-x-4 text-xs text-muted-foreground">
            <span>Version 1.0</span>
            <span>•</span>
            <span>Status: Active</span>
            <span>•</span>
            <span className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-status-normal rounded-full"></div>
              <span>System Online</span>
            </span>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;