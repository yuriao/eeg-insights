import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Nav from './components/Nav'
import Home from './pages/Home'
import Post from './pages/Post'
import Algorithms from './pages/Algorithms'
import CustomAnalysis from './pages/CustomAnalysis'
import './index.css'

export default function App() {
  return (
    <BrowserRouter basename="/eeg-insights">
      <Nav />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/post/:slug" element={<Post />} />
        <Route path="/algorithms" element={<Algorithms />} />
        <Route path="/custom" element={<CustomAnalysis />} />
      </Routes>
    </BrowserRouter>
  )
}
