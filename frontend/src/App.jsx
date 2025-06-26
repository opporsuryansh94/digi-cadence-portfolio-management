import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  BarChart3, 
  TrendingUp, 
  Users, 
  Target, 
  Brain, 
  Shield, 
  Zap,
  Settings,
  Bell,
  Search,
  Menu,
  X,
  ChevronDown,
  Activity,
  PieChart,
  LineChart,
  Globe,
  Briefcase,
  Star,
  AlertTriangle,
  CheckCircle,
  Clock,
  ArrowUpRight,
  ArrowDownRight,
  Minus
} from 'lucide-react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar.jsx'
import { Input } from '@/components/ui/input.jsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuLabel, 
  DropdownMenuSeparator, 
  DropdownMenuTrigger 
} from '@/components/ui/dropdown-menu.jsx'
import { 
  LineChart as RechartsLineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  BarChart as RechartsBarChart,
  Bar,
  PieChart as RechartsPieChart,
  Cell,
  Pie,
  Area,
  AreaChart
} from 'recharts'
import './App.css'

// Mock data for demonstration
const portfolioData = {
  overview: {
    totalBrands: 24,
    totalProjects: 8,
    totalOrganizations: 3,
    portfolioHealthScore: 0.87,
    totalRevenue: 12500000,
    avgROI: 3.4,
    marketShare: 0.18
  },
  brands: [
    { id: 1, name: 'TechFlow', category: 'Technology', healthScore: 0.92, revenue: 2800000, roi: 4.2, trend: 'up' },
    { id: 2, name: 'EcoVibe', category: 'Sustainability', healthScore: 0.85, revenue: 1900000, roi: 3.8, trend: 'up' },
    { id: 3, name: 'UrbanStyle', category: 'Fashion', healthScore: 0.78, revenue: 1600000, roi: 2.9, trend: 'stable' },
    { id: 4, name: 'FoodCraft', category: 'Food & Beverage', healthScore: 0.71, revenue: 1400000, roi: 2.1, trend: 'down' },
    { id: 5, name: 'HealthPlus', category: 'Healthcare', healthScore: 0.89, revenue: 2200000, roi: 3.9, trend: 'up' },
    { id: 6, name: 'EduTech', category: 'Education', healthScore: 0.83, revenue: 1800000, roi: 3.2, trend: 'up' }
  ],
  performanceData: [
    { month: 'Jan', revenue: 980000, roi: 3.1, brandAwareness: 0.65, marketShare: 0.16 },
    { month: 'Feb', revenue: 1050000, roi: 3.2, brandAwareness: 0.68, marketShare: 0.165 },
    { month: 'Mar', revenue: 1120000, roi: 3.4, brandAwareness: 0.71, marketShare: 0.17 },
    { month: 'Apr', revenue: 1080000, roi: 3.3, brandAwareness: 0.69, marketShare: 0.168 },
    { month: 'May', revenue: 1180000, roi: 3.6, brandAwareness: 0.73, marketShare: 0.175 },
    { month: 'Jun', revenue: 1250000, roi: 3.8, brandAwareness: 0.76, marketShare: 0.18 }
  ],
  agentStatus: [
    { name: 'Portfolio Optimizer', status: 'active', tasksCompleted: 156, efficiency: 0.94 },
    { name: 'Brand Metric Optimizer', status: 'active', tasksCompleted: 203, efficiency: 0.91 },
    { name: 'Forecasting Agent', status: 'active', tasksCompleted: 89, efficiency: 0.96 },
    { name: 'Strategy Agent', status: 'active', tasksCompleted: 67, efficiency: 0.88 }
  ],
  insights: [
    { type: 'opportunity', title: 'Cross-Brand Synergy Detected', description: 'TechFlow and EduTech show 85% correlation in customer segments', priority: 'high' },
    { type: 'warning', title: 'FoodCraft Performance Decline', description: 'ROI decreased by 15% over last quarter', priority: 'high' },
    { type: 'success', title: 'HealthPlus Market Expansion', description: 'Successfully captured 12% additional market share', priority: 'medium' },
    { type: 'info', title: 'Seasonal Trend Identified', description: 'Q2 shows consistent 18% performance boost across portfolio', priority: 'low' }
  ]
}

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4']

function Dashboard() {
  const [selectedTab, setSelectedTab] = useState('overview')
  const [sidebarOpen, setSidebarOpen] = useState(true)

  const getTrendIcon = (trend) => {
    switch (trend) {
      case 'up': return <ArrowUpRight className="h-4 w-4 text-green-500" />
      case 'down': return <ArrowDownRight className="h-4 w-4 text-red-500" />
      default: return <Minus className="h-4 w-4 text-gray-500" />
    }
  }

  const getInsightIcon = (type) => {
    switch (type) {
      case 'opportunity': return <TrendingUp className="h-4 w-4 text-green-500" />
      case 'warning': return <AlertTriangle className="h-4 w-4 text-yellow-500" />
      case 'success': return <CheckCircle className="h-4 w-4 text-green-500" />
      default: return <Activity className="h-4 w-4 text-blue-500" />
    }
  }

  const getPriorityBadge = (priority) => {
    const variants = {
      high: 'destructive',
      medium: 'default',
      low: 'secondary'
    }
    return <Badge variant={variants[priority]}>{priority}</Badge>
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-card/50 backdrop-blur supports-[backdrop-filter]:bg-card/50">
        <div className="flex h-16 items-center px-6">
          <div className="flex items-center space-x-4">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="md:hidden"
            >
              {sidebarOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
            </Button>
            <div className="flex items-center space-x-2">
              <div className="h-8 w-8 rounded-lg bg-primary flex items-center justify-center">
                <Brain className="h-5 w-5 text-primary-foreground" />
              </div>
              <h1 className="text-xl font-bold">Digi-Cadence</h1>
            </div>
          </div>
          
          <div className="ml-auto flex items-center space-x-4">
            <div className="relative hidden md:block">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Search brands, projects..."
                className="w-64 pl-9"
              />
            </div>
            <Button variant="ghost" size="icon">
              <Bell className="h-4 w-4" />
            </Button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" className="relative h-8 w-8 rounded-full">
                  <Avatar className="h-8 w-8">
                    <AvatarImage src="/avatars/01.png" alt="@user" />
                    <AvatarFallback>CM</AvatarFallback>
                  </Avatar>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-56" align="end" forceMount>
                <DropdownMenuLabel className="font-normal">
                  <div className="flex flex-col space-y-1">
                    <p className="text-sm font-medium leading-none">Chief Marketing Officer</p>
                    <p className="text-xs leading-none text-muted-foreground">cmo@company.com</p>
                  </div>
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem>
                  <Settings className="mr-2 h-4 w-4" />
                  <span>Settings</span>
                </DropdownMenuItem>
                <DropdownMenuItem>
                  <span>Sign out</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar */}
        <AnimatePresence>
          {sidebarOpen && (
            <motion.aside
              initial={{ x: -300 }}
              animate={{ x: 0 }}
              exit={{ x: -300 }}
              transition={{ duration: 0.3 }}
              className="w-64 border-r bg-card/50 backdrop-blur supports-[backdrop-filter]:bg-card/50 h-[calc(100vh-4rem)]"
            >
              <nav className="p-4 space-y-2">
                <Button
                  variant={selectedTab === 'overview' ? 'default' : 'ghost'}
                  className="w-full justify-start"
                  onClick={() => setSelectedTab('overview')}
                >
                  <BarChart3 className="mr-2 h-4 w-4" />
                  Portfolio Overview
                </Button>
                <Button
                  variant={selectedTab === 'brands' ? 'default' : 'ghost'}
                  className="w-full justify-start"
                  onClick={() => setSelectedTab('brands')}
                >
                  <Target className="mr-2 h-4 w-4" />
                  Brand Management
                </Button>
                <Button
                  variant={selectedTab === 'analytics' ? 'default' : 'ghost'}
                  className="w-full justify-start"
                  onClick={() => setSelectedTab('analytics')}
                >
                  <TrendingUp className="mr-2 h-4 w-4" />
                  Advanced Analytics
                </Button>
                <Button
                  variant={selectedTab === 'agents' ? 'default' : 'ghost'}
                  className="w-full justify-start"
                  onClick={() => setSelectedTab('agents')}
                >
                  <Brain className="mr-2 h-4 w-4" />
                  AI Agents
                </Button>
                <Button
                  variant={selectedTab === 'reports' ? 'default' : 'ghost'}
                  className="w-full justify-start"
                  onClick={() => setSelectedTab('reports')}
                >
                  <PieChart className="mr-2 h-4 w-4" />
                  Reports
                </Button>
                <Button
                  variant={selectedTab === 'insights' ? 'default' : 'ghost'}
                  className="w-full justify-start"
                  onClick={() => setSelectedTab('insights')}
                >
                  <Zap className="mr-2 h-4 w-4" />
                  Insights
                </Button>
              </nav>
            </motion.aside>
          )}
        </AnimatePresence>

        {/* Main Content */}
        <main className="flex-1 p-6 overflow-auto">
          <AnimatePresence mode="wait">
            <motion.div
              key={selectedTab}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              {selectedTab === 'overview' && (
                <div className="space-y-6">
                  <div>
                    <h2 className="text-3xl font-bold tracking-tight">Portfolio Overview</h2>
                    <p className="text-muted-foreground">
                      Comprehensive view of your brand portfolio performance
                    </p>
                  </div>

                  {/* Key Metrics */}
                  <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                    <Card>
                      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Total Brands</CardTitle>
                        <Target className="h-4 w-4 text-muted-foreground" />
                      </CardHeader>
                      <CardContent>
                        <div className="text-2xl font-bold">{portfolioData.overview.totalBrands}</div>
                        <p className="text-xs text-muted-foreground">
                          Across {portfolioData.overview.totalProjects} projects
                        </p>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Portfolio Health</CardTitle>
                        <Activity className="h-4 w-4 text-muted-foreground" />
                      </CardHeader>
                      <CardContent>
                        <div className="text-2xl font-bold">
                          {(portfolioData.overview.portfolioHealthScore * 100).toFixed(0)}%
                        </div>
                        <Progress value={portfolioData.overview.portfolioHealthScore * 100} className="mt-2" />
                      </CardContent>
                    </Card>
                    <Card>
                      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Total Revenue</CardTitle>
                        <TrendingUp className="h-4 w-4 text-muted-foreground" />
                      </CardHeader>
                      <CardContent>
                        <div className="text-2xl font-bold">
                          ${(portfolioData.overview.totalRevenue / 1000000).toFixed(1)}M
                        </div>
                        <p className="text-xs text-muted-foreground">
                          +12% from last quarter
                        </p>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Average ROI</CardTitle>
                        <BarChart3 className="h-4 w-4 text-muted-foreground" />
                      </CardHeader>
                      <CardContent>
                        <div className="text-2xl font-bold">{portfolioData.overview.avgROI}x</div>
                        <p className="text-xs text-muted-foreground">
                          Above industry average
                        </p>
                      </CardContent>
                    </Card>
                  </div>

                  {/* Performance Chart */}
                  <Card>
                    <CardHeader>
                      <CardTitle>Portfolio Performance Trends</CardTitle>
                      <CardDescription>
                        Revenue, ROI, and market metrics over time
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="h-80">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={portfolioData.performanceData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="month" />
                            <YAxis />
                            <Tooltip />
                            <Area 
                              type="monotone" 
                              dataKey="revenue" 
                              stackId="1" 
                              stroke="#3b82f6" 
                              fill="#3b82f6" 
                              fillOpacity={0.6}
                            />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Top Performing Brands */}
                  <Card>
                    <CardHeader>
                      <CardTitle>Top Performing Brands</CardTitle>
                      <CardDescription>
                        Brands ranked by health score and performance
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        {portfolioData.brands.slice(0, 3).map((brand, index) => (
                          <div key={brand.id} className="flex items-center space-x-4">
                            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10">
                              <span className="text-sm font-medium">#{index + 1}</span>
                            </div>
                            <div className="flex-1 space-y-1">
                              <div className="flex items-center justify-between">
                                <p className="text-sm font-medium leading-none">{brand.name}</p>
                                <div className="flex items-center space-x-1">
                                  {getTrendIcon(brand.trend)}
                                  <span className="text-sm text-muted-foreground">
                                    {(brand.healthScore * 100).toFixed(0)}%
                                  </span>
                                </div>
                              </div>
                              <p className="text-sm text-muted-foreground">{brand.category}</p>
                              <Progress value={brand.healthScore * 100} className="h-2" />
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </div>
              )}

              {selectedTab === 'brands' && (
                <div className="space-y-6">
                  <div>
                    <h2 className="text-3xl font-bold tracking-tight">Brand Management</h2>
                    <p className="text-muted-foreground">
                      Detailed view and management of individual brand performance
                    </p>
                  </div>

                  <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                    {portfolioData.brands.map((brand) => (
                      <Card key={brand.id} className="hover:shadow-lg transition-shadow">
                        <CardHeader>
                          <div className="flex items-center justify-between">
                            <CardTitle className="text-lg">{brand.name}</CardTitle>
                            {getTrendIcon(brand.trend)}
                          </div>
                          <CardDescription>{brand.category}</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-4">
                          <div>
                            <div className="flex justify-between text-sm">
                              <span>Health Score</span>
                              <span>{(brand.healthScore * 100).toFixed(0)}%</span>
                            </div>
                            <Progress value={brand.healthScore * 100} className="mt-1" />
                          </div>
                          <div className="grid grid-cols-2 gap-4 text-sm">
                            <div>
                              <p className="text-muted-foreground">Revenue</p>
                              <p className="font-medium">${(brand.revenue / 1000000).toFixed(1)}M</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">ROI</p>
                              <p className="font-medium">{brand.roi}x</p>
                            </div>
                          </div>
                          <Button className="w-full" variant="outline">
                            View Details
                          </Button>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </div>
              )}

              {selectedTab === 'analytics' && (
                <div className="space-y-6">
                  <div>
                    <h2 className="text-3xl font-bold tracking-tight">Advanced Analytics</h2>
                    <p className="text-muted-foreground">
                      AI-powered analytics and insights across your portfolio
                    </p>
                  </div>

                  <div className="grid gap-6 md:grid-cols-2">
                    <Card>
                      <CardHeader>
                        <CardTitle>Brand Performance Distribution</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="h-64">
                          <ResponsiveContainer width="100%" height="100%">
                            <RechartsPieChart>
                              <Pie
                                data={portfolioData.brands.map(brand => ({
                                  name: brand.name,
                                  value: brand.revenue
                                }))}
                                cx="50%"
                                cy="50%"
                                outerRadius={80}
                                fill="#8884d8"
                                dataKey="value"
                              >
                                {portfolioData.brands.map((entry, index) => (
                                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                ))}
                              </Pie>
                              <Tooltip formatter={(value) => `$${(value / 1000000).toFixed(1)}M`} />
                            </RechartsPieChart>
                          </ResponsiveContainer>
                        </div>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader>
                        <CardTitle>ROI vs Health Score</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="h-64">
                          <ResponsiveContainer width="100%" height="100%">
                            <RechartsBarChart data={portfolioData.brands}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="name" />
                              <YAxis />
                              <Tooltip />
                              <Bar dataKey="roi" fill="#3b82f6" />
                            </RechartsBarChart>
                          </ResponsiveContainer>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </div>
              )}

              {selectedTab === 'agents' && (
                <div className="space-y-6">
                  <div>
                    <h2 className="text-3xl font-bold tracking-tight">AI Agent System</h2>
                    <p className="text-muted-foreground">
                      Monitor and manage your intelligent agent ecosystem
                    </p>
                  </div>

                  <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                    {portfolioData.agentStatus.map((agent, index) => (
                      <Card key={index}>
                        <CardHeader className="pb-2">
                          <div className="flex items-center justify-between">
                            <CardTitle className="text-sm">{agent.name}</CardTitle>
                            <Badge variant={agent.status === 'active' ? 'default' : 'secondary'}>
                              {agent.status}
                            </Badge>
                          </div>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                              <span>Efficiency</span>
                              <span>{(agent.efficiency * 100).toFixed(0)}%</span>
                            </div>
                            <Progress value={agent.efficiency * 100} />
                            <p className="text-xs text-muted-foreground">
                              {agent.tasksCompleted} tasks completed
                            </p>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>

                  <Card>
                    <CardHeader>
                      <CardTitle>Agent Performance Metrics</CardTitle>
                      <CardDescription>
                        Real-time performance monitoring of AI agents
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                          <RechartsLineChart data={portfolioData.agentStatus.map(agent => ({
                            name: agent.name.split(' ')[0],
                            efficiency: agent.efficiency * 100,
                            tasks: agent.tasksCompleted
                          }))}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="name" />
                            <YAxis />
                            <Tooltip />
                            <Line type="monotone" dataKey="efficiency" stroke="#3b82f6" strokeWidth={2} />
                          </RechartsLineChart>
                        </ResponsiveContainer>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              )}

              {selectedTab === 'insights' && (
                <div className="space-y-6">
                  <div>
                    <h2 className="text-3xl font-bold tracking-tight">AI-Powered Insights</h2>
                    <p className="text-muted-foreground">
                      Intelligent recommendations and actionable insights
                    </p>
                  </div>

                  <div className="space-y-4">
                    {portfolioData.insights.map((insight, index) => (
                      <Card key={index} className="hover:shadow-md transition-shadow">
                        <CardContent className="pt-6">
                          <div className="flex items-start space-x-4">
                            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10">
                              {getInsightIcon(insight.type)}
                            </div>
                            <div className="flex-1 space-y-2">
                              <div className="flex items-center justify-between">
                                <h3 className="font-semibold">{insight.title}</h3>
                                {getPriorityBadge(insight.priority)}
                              </div>
                              <p className="text-sm text-muted-foreground">{insight.description}</p>
                              <div className="flex space-x-2">
                                <Button size="sm" variant="outline">
                                  View Details
                                </Button>
                                <Button size="sm">
                                  Take Action
                                </Button>
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </div>
              )}

              {selectedTab === 'reports' && (
                <div className="space-y-6">
                  <div>
                    <h2 className="text-3xl font-bold tracking-tight">Comprehensive Reports</h2>
                    <p className="text-muted-foreground">
                      Generate and access detailed portfolio reports
                    </p>
                  </div>

                  <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                    {[
                      { name: 'Portfolio Performance', description: 'Comprehensive portfolio analysis', icon: BarChart3 },
                      { name: 'Brand Equity Analysis', description: 'Brand equity measurement and trends', icon: Target },
                      { name: 'Competitive Intelligence', description: 'Market positioning and competitor analysis', icon: Globe },
                      { name: 'Digital Marketing Effectiveness', description: 'Campaign and channel performance', icon: TrendingUp },
                      { name: 'Cross-Brand Synergy', description: 'Synergy identification and optimization', icon: Users },
                      { name: 'Predictive Insights', description: 'Forecasting and predictive analytics', icon: Brain }
                    ].map((report, index) => (
                      <Card key={index} className="hover:shadow-lg transition-shadow cursor-pointer">
                        <CardHeader>
                          <div className="flex items-center space-x-2">
                            <report.icon className="h-5 w-5 text-primary" />
                            <CardTitle className="text-lg">{report.name}</CardTitle>
                          </div>
                          <CardDescription>{report.description}</CardDescription>
                        </CardHeader>
                        <CardContent>
                          <Button className="w-full">
                            Generate Report
                          </Button>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  )
}

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  )
}

export default App

