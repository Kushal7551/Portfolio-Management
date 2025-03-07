import React from 'react';
const LoginPage = () => {
  return (
    <div className="flex h-screen">
      {/* Background Image */}
      <div className="flex-1 bg-cover bg-center" style={{ backgroundImage: "url('/api/placeholder/1280/832')" }}></div>

      {/* Login Form */}
      <div className="w-1/2 bg-white p-12 flex flex-col justify-center">
        <div className="max-w-md w-full mx-auto">
          <div className="flex justify-center mb-8">
            {/* Simple shield icon using SVG */}
            <svg xmlns="src/assets/svg" className="w-16 h-16 text-green-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
            </svg>
          </div>
          <h2 className="text-3xl font-bold text-center mb-2">CREATE AN ACCOUNT</h2>
          <p className="text-center text-gray-600 mb-8">Join KisanDost!</p>
          <form>
            <div className="space-y-4">
              <input type="text" placeholder="Username" className="w-full px-3 py-2 border rounded-md" />
              <input type="email" placeholder="Email id" className="w-full px-3 py-2 border rounded-md" />
              <input type="password" placeholder="Password" className="w-full px-3 py-2 border rounded-md" />
              <input type="password" placeholder="Confirm Password" className="w-full px-3 py-2 border rounded-md" />
            </div>
            <button type="submit" className="w-full bg-green-500 hover:bg-green-600 text-white py-2 rounded-md mt-6">
              Sign Up
              <span className="ml-2"></span>
            </button>
          </form>
          <p className="text-center text-sm mt-6">
            Already have an account? <a href="#" className="text-green-500 hover:underline">LOG IN</a>
          </p>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;