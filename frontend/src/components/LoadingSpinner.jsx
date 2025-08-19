import React from 'react';

const LoadingSpinner = ({ size = 'medium', message = 'Chargement...' }) => {
    const sizeClasses = {
        small: 'h-6 w-6',
        medium: 'h-10 w-10',
        large: 'h-16 w-16'
    };

    const textSizeClasses = {
        small: 'text-sm',
        medium: 'text-base',
        large: 'text-lg'
    };

    return (
        <div className="flex flex-col items-center justify-center py-8">
            <div className={`animate-spin rounded-full border-b-2 border-blue-600 ${sizeClasses[size]} mb-4`}></div>
            <p className={`text-gray-600 ${textSizeClasses[size]}`}>{message}</p>
        </div>
    );
};

export default LoadingSpinner;