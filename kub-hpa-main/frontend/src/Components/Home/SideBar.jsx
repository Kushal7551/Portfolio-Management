import React, { useState } from 'react';
import { Link } from 'react-router-dom';

function App() {
    const [isOpen, setIsOpen] = useState(false);

    const toggleSidebar = () => {
        setIsOpen(!isOpen);
    };

    return (
        <div className="flex min-h-screen bg-gray-100">
            {/* Sidebar */}
            <div className={`fixed inset-y-0 left-0 transform ${isOpen ? 'translate-x-0' : '-translate-x-full'} md:relative md:translate-x-0 transition-transform duration-300 ease-in-out bg-gray-800 text-white w-64 p-4`}>
                <nav>
                    <ul>
                        {/* <li className="mb-4"> */}
                            {/* <Link to="stock-analysis" className="text-lg hover:text-gray-400">Stock Analysis</Link> */}
                        {/* </li> */}
                        <li className="mb-4">
                            <Link to="algorithmic-trading" className="text-lg hover:text-gray-400">Algorithmic Trading</Link>
                        </li>
                        {/* <li className="mb-4"> */}
                            {/* <Link to="stock-prices" className="text-lg hover:text-gray-400">Stock Prices</Link> */}
                        {/* </li> */}
                        {/* <li className="mb-4"> */}
                            {/* <Link to="about" className="text-lg hover:text-gray-400"> News </Link> */}
                        {/* </li> */}
                    </ul>
                </nav>
            </div>

        </div>
    );
}

export default App;
