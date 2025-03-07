import React from 'react'
import Navbar from '../Navbar'
import Header from '../Header'
import { useNavigate } from 'react-router-dom';
import Sidebar from './SideBar';
import News from '../Live-News/News';


// import FarmCard from '../Farms/FarmCard'

const Home = () => {
  const navigate = useNavigate();
  return (
    <div>
        <Navbar/>
        <Header text="HOME" />
        <div className="flex min-h-screen">
      <Sidebar />
      <News />
      <News />
    </div>
    </div>
  )
}

export default Home
