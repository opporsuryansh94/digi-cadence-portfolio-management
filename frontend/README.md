# Digi-Cadence Frontend

**🎨 Modern React Dashboard with Real-time Analytics**

## Overview

The Digi-Cadence frontend is a sophisticated React application that provides a professional, responsive interface for the portfolio management platform with real-time data visualization, interactive dashboards, and mobile optimization.

## Features

### Dashboard Components
- **Portfolio Overview**: Executive dashboard with KPIs and performance trends
- **Brand Management**: Individual brand cards with health scores and metrics
- **Advanced Analytics**: Interactive charts with comprehensive data visualization
- **AI Agent Monitoring**: Real-time agent status and performance tracking
- **Comprehensive Reports**: Interface for all 16 report types
- **AI-Powered Insights**: Intelligent recommendations with priority indicators

### Technical Features
- **React 18**: Latest React with concurrent features
- **Responsive Design**: Mobile-first design optimized for all devices
- **Real-time Updates**: WebSocket integration for live data
- **Interactive Charts**: Recharts integration with dynamic visualizations
- **Modern UI**: shadcn/ui components with Tailwind CSS
- **Performance Optimized**: Code splitting and lazy loading

## Technology Stack

- **React 18**: Modern React with hooks and concurrent features
- **Vite**: Fast build tool and development server
- **Tailwind CSS**: Utility-first CSS framework
- **shadcn/ui**: High-quality React components
- **Recharts**: Composable charting library
- **Framer Motion**: Smooth animations and transitions
- **Lucide React**: Beautiful icon library

## Installation

### Prerequisites

- Node.js 18+
- npm or yarn

### Setup

1. **Install dependencies**
   ```bash
   npm install
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API endpoint
   ```

3. **Start development server**
   ```bash
   npm run dev
   ```

4. **Build for production**
   ```bash
   npm run build
   ```

## Project Structure

```
src/
├── App.jsx                 # Main application component
├── App.css                 # Global styles and responsive design
├── components/             # Reusable components
├── pages/                  # Page components
├── hooks/                  # Custom React hooks
├── utils/                  # Utility functions
├── services/               # API service functions
└── assets/                 # Static assets
```

## Key Components

### Dashboard Layout
```jsx
// Main dashboard with sidebar navigation
const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('overview');
  
  return (
    <div className="dashboard-layout">
      <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />
      <MainContent activeTab={activeTab} />
    </div>
  );
};
```

### Portfolio Overview
```jsx
// Executive dashboard with KPIs
const PortfolioOverview = () => {
  const [portfolioData, setPortfolioData] = useState(null);
  
  useEffect(() => {
    fetchPortfolioData().then(setPortfolioData);
  }, []);
  
  return (
    <div className="portfolio-overview">
      <KPICards data={portfolioData} />
      <PerformanceCharts data={portfolioData} />
      <InsightsPanel data={portfolioData} />
    </div>
  );
};
```

### Brand Management
```jsx
// Brand cards with health scores
const BrandManagement = () => {
  const [brands, setBrands] = useState([]);
  
  return (
    <div className="brand-grid">
      {brands.map(brand => (
        <BrandCard key={brand.id} brand={brand} />
      ))}
    </div>
  );
};
```

## Responsive Design

### Mobile Optimization
- **Touch-friendly interface**: Optimized for touch interactions
- **Responsive layouts**: Adapts to all screen sizes
- **Mobile navigation**: Collapsible sidebar for mobile devices
- **Optimized performance**: Fast loading on mobile networks

### Breakpoints
```css
/* Tailwind CSS breakpoints */
sm: 640px   /* Small devices */
md: 768px   /* Medium devices */
lg: 1024px  /* Large devices */
xl: 1280px  /* Extra large devices */
2xl: 1536px /* 2X large devices */
```

## Data Visualization

### Chart Types
- **Line Charts**: Trend analysis and time-series data
- **Bar Charts**: Comparative analysis and metrics
- **Pie Charts**: Portfolio composition and distribution
- **Area Charts**: Cumulative metrics and performance
- **Scatter Plots**: Correlation analysis and positioning

### Interactive Features
- **Zoom and Pan**: Detailed data exploration
- **Tooltips**: Contextual information on hover
- **Filtering**: Dynamic data filtering and segmentation
- **Drill-down**: Navigate from summary to detailed views

## API Integration

### Service Layer
```javascript
// API service for backend communication
class ApiService {
  constructor(baseURL) {
    this.baseURL = baseURL;
    this.token = localStorage.getItem('authToken');
  }
  
  async fetchPortfolioData() {
    const response = await fetch(`${this.baseURL}/api/portfolio`, {
      headers: { Authorization: `Bearer ${this.token}` }
    });
    return response.json();
  }
  
  async generateReport(reportType, options) {
    const response = await fetch(`${this.baseURL}/api/reports/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.token}`
      },
      body: JSON.stringify({ reportType, options })
    });
    return response.json();
  }
}
```

### Real-time Updates
```javascript
// WebSocket integration for real-time data
const useWebSocket = (url) => {
  const [data, setData] = useState(null);
  
  useEffect(() => {
    const ws = new WebSocket(url);
    
    ws.onmessage = (event) => {
      const newData = JSON.parse(event.data);
      setData(newData);
    };
    
    return () => ws.close();
  }, [url]);
  
  return data;
};
```

## Performance Optimization

### Code Splitting
```javascript
// Lazy loading for better performance
const AnalyticsPage = lazy(() => import('./pages/Analytics'));
const ReportsPage = lazy(() => import('./pages/Reports'));

// Usage with Suspense
<Suspense fallback={<LoadingSpinner />}>
  <AnalyticsPage />
</Suspense>
```

### Memoization
```javascript
// Optimize expensive calculations
const ExpensiveComponent = memo(({ data }) => {
  const processedData = useMemo(() => {
    return processLargeDataset(data);
  }, [data]);
  
  return <Chart data={processedData} />;
});
```

## Testing

### Unit Tests
```bash
# Run unit tests
npm test

# Run tests with coverage
npm run test:coverage
```

### E2E Tests
```bash
# Run end-to-end tests
npm run test:e2e
```

## Build and Deployment

### Development Build
```bash
npm run dev
```

### Production Build
```bash
npm run build
npm run preview
```

### Docker Deployment
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "run", "preview"]
```

## Environment Configuration

```bash
# .env file
VITE_API_BASE_URL=http://localhost:5000
VITE_WS_URL=ws://localhost:5000/ws
VITE_APP_NAME=Digi-Cadence
VITE_APP_VERSION=1.0.0
```

## Accessibility

- **ARIA Labels**: Proper accessibility labels
- **Keyboard Navigation**: Full keyboard support
- **Screen Reader Support**: Compatible with screen readers
- **Color Contrast**: WCAG 2.1 AA compliant
- **Focus Management**: Proper focus handling

## Browser Support

- **Chrome**: 90+
- **Firefox**: 88+
- **Safari**: 14+
- **Edge**: 90+

## Contributing

1. Follow React best practices
2. Use TypeScript for type safety
3. Write comprehensive tests
4. Follow accessibility guidelines
5. Optimize for performance
6. Update documentation

## License

MIT License - see LICENSE file for details.

