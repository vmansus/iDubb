import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { ThemeProvider } from './contexts/ThemeContext'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import NewTask from './pages/NewTask'
import TaskDetail from './pages/TaskDetail'
import Settings from './pages/Settings'
import Presets from './pages/Presets'
import Subscriptions from './pages/Subscriptions'
import Discover from './pages/Discover'
import Glossary from './pages/Glossary'

function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="new" element={<NewTask />} />
            <Route path="task/:taskId" element={<TaskDetail />} />
            <Route path="discover" element={<Discover />} />
            <Route path="subscriptions" element={<Subscriptions />} />
            <Route path="settings" element={<Settings />} />
            <Route path="presets" element={<Presets />} />
            <Route path="glossary" element={<Glossary />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  )
}

export default App
