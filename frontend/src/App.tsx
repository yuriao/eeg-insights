import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Nav from './components/Nav'
import Home from './pages/Home'
import Post from './pages/Post'
import './index.css'

export default function App() {
  return (
    <BrowserRouter basename="/eeg-insights">
      <Nav />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/post/:slug" element={<Post />} />
      </Routes>
    </BrowserRouter>
  )
}
