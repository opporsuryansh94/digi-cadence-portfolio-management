# Digi-Cadence Backend

**🚀 Enterprise Flask Application with Advanced Analytics and AI**

## Overview

The Digi-Cadence backend is a sophisticated Flask application that provides the core functionality for the portfolio management platform, including advanced analytics, multi-agent AI system, and comprehensive API endpoints.

## Architecture

### Core Components

- **Flask Application**: Main web application with factory pattern
- **PostgreSQL Database**: Multi-tenant database with comprehensive schema
- **Redis Cache**: Session management and caching layer
- **MCP Servers**: Distributed processing architecture
- **Multi-Agent System**: Four autonomous AI agents
- **Analytics Engine**: Advanced analytics with machine learning

### Directory Structure

```
src/
├── main.py                 # Application entry point
├── config.py              # Configuration management
├── models/
│   └── portfolio.py       # Database models
├── routes/
│   ├── portfolio.py       # Portfolio management endpoints
│   ├── analytics.py       # Analytics endpoints
│   ├── agents.py          # Agent management endpoints
│   ├── mcp.py            # MCP server endpoints
│   └── reports.py        # Reporting endpoints
├── analytics/
│   ├── genetic_optimizer.py      # Genetic algorithm optimization
│   ├── shap_analyzer.py          # SHAP analysis
│   ├── correlation_analyzer.py   # Correlation analysis
│   ├── competitive_gap_analyzer.py # Competitive analysis
│   └── trend_analyzer.py         # Trend analysis and forecasting
├── agents/
│   ├── base_agent.py                          # Base agent framework
│   ├── portfolio_optimization_agent.py       # Portfolio optimization
│   ├── multi_brand_metric_optimization_agent.py # Multi-brand optimization
│   ├── portfolio_forecasting_agent.py        # Forecasting agent
│   └── portfolio_strategy_agent.py           # Strategy agent
├── mcp_servers/
│   ├── base_server.py     # Base MCP server
│   └── analysis_server.py # Analysis MCP server
├── reports/
│   ├── base_generator.py  # Base report generator
│   └── generators/        # Specific report generators
└── security/
    └── security_manager.py # Security implementation
```

## Installation

### Prerequisites

- Python 3.11+
- PostgreSQL 13+
- Redis 6+

### Setup

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configurations
   ```

4. **Initialize database**
   ```bash
   flask db init
   flask db migrate
   flask db upgrade
   ```

5. **Run application**
   ```bash
   python src/main.py
   ```

## API Endpoints

### Portfolio Management
- `GET /api/organizations` - List organizations
- `POST /api/organizations` - Create organization
- `GET /api/projects` - List projects
- `POST /api/projects` - Create project
- `GET /api/brands` - List brands
- `POST /api/brands` - Create brand

### Analytics
- `POST /api/analytics/optimize` - Portfolio optimization
- `POST /api/analytics/shap` - SHAP analysis
- `POST /api/analytics/correlations` - Correlation analysis
- `POST /api/analytics/gaps` - Competitive gap analysis
- `POST /api/analytics/trends` - Trend analysis

### Agents
- `GET /api/agents` - List agents
- `POST /api/agents/{agent_id}/start` - Start agent
- `POST /api/agents/{agent_id}/stop` - Stop agent
- `GET /api/agents/{agent_id}/status` - Agent status

### Reports
- `GET /api/reports/types` - Available report types
- `POST /api/reports/generate` - Generate report
- `GET /api/reports/{report_id}` - Get report
- `GET /api/reports/{report_id}/download` - Download report

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:port/dbname

# Redis
REDIS_URL=redis://host:port/0

# Security
SECRET_KEY=your-secret-key
JWT_SECRET_KEY=your-jwt-secret
ENCRYPTION_KEY=your-encryption-key

# Application
FLASK_ENV=production
FLASK_DEBUG=False
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
python -m pytest tests/security/ -v
```

## Security Features

- **JWT Authentication**: Secure API access
- **Role-based Access Control**: Granular permissions
- **AES-256 Encryption**: Data encryption
- **Input Validation**: SQL injection protection
- **Audit Logging**: Comprehensive audit trail
- **Rate Limiting**: API rate limiting
- **CSRF Protection**: Cross-site request forgery protection

## Performance

- **Database Connection Pooling**: Optimized database connections
- **Redis Caching**: Fast data retrieval
- **Async Processing**: Non-blocking operations
- **Query Optimization**: Efficient database queries
- **Resource Management**: Memory and CPU optimization

## Monitoring

- **Health Checks**: Application health monitoring
- **Performance Metrics**: Response time and throughput
- **Error Tracking**: Comprehensive error logging
- **Resource Usage**: CPU and memory monitoring

## Deployment

### Production Configuration

```python
# config.py
class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    DATABASE_URL = os.environ.get('DATABASE_URL')
    REDIS_URL = os.environ.get('REDIS_URL')
    SECRET_KEY = os.environ.get('SECRET_KEY')
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
EXPOSE 5000
CMD ["python", "src/main.py"]
```

## Contributing

1. Follow PEP 8 style guidelines
2. Write comprehensive tests
3. Update documentation
4. Ensure security best practices
5. Add type hints where appropriate

## License

MIT License - see LICENSE file for details.

