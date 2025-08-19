import React from 'react';

const Header = () => {
    return (
        // Header
        <header className="bg-white shadow-sm border-b border-gray-200">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                            <img src="/quantys.png" alt="Quantys Logo" className="h-12 w-auto" />
                            <div>
                                <h1 className="text-2xl font-bold text-gray-900">Moulinette Sage X3</h1>
                                <p className="text-sm text-gray-500">Automatisation des traitements d'inventaire</p>
                            </div>
                        </div>
                        <div className="flex items-center space-x-2">
                            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                                Version 1.0
                            </span>
                        </div>
                    </div>
                </div>
        </header>
    );
};

export default Header;