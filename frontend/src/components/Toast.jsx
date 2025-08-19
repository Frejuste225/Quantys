import React, { useState, useEffect } from 'react';
import { CheckCircle, AlertCircle, X, Info } from 'lucide-react';

const Toast = ({ type = 'info', message, onClose, duration = 5000 }) => {
    const [isVisible, setIsVisible] = useState(true);

    useEffect(() => {
        const timer = setTimeout(() => {
            setIsVisible(false);
            setTimeout(onClose, 300); // Attendre la fin de l'animation
        }, duration);

        return () => clearTimeout(timer);
    }, [duration, onClose]);

    const handleClose = () => {
        setIsVisible(false);
        setTimeout(onClose, 300);
    };

    const typeConfig = {
        success: {
            icon: CheckCircle,
            bgColor: 'bg-green-100',
            borderColor: 'border-green-400',
            textColor: 'text-green-700',
            iconColor: 'text-green-600'
        },
        error: {
            icon: AlertCircle,
            bgColor: 'bg-red-100',
            borderColor: 'border-red-400',
            textColor: 'text-red-700',
            iconColor: 'text-red-600'
        },
        info: {
            icon: Info,
            bgColor: 'bg-blue-100',
            borderColor: 'border-blue-400',
            textColor: 'text-blue-700',
            iconColor: 'text-blue-600'
        }
    };

    const config = typeConfig[type];
    const Icon = config.icon;

    return (
        <div className={`fixed top-4 right-4 z-50 transition-all duration-300 transform ${
            isVisible ? 'translate-x-0 opacity-100' : 'translate-x-full opacity-0'
        }`}>
            <div className={`${config.bgColor} ${config.borderColor} ${config.textColor} border rounded-lg p-4 shadow-lg max-w-sm`}>
                <div className="flex items-start">
                    <Icon className={`h-5 w-5 ${config.iconColor} mr-3 mt-0.5 flex-shrink-0`} />
                    <div className="flex-1">
                        <p className="text-sm font-medium">{message}</p>
                    </div>
                    <button
                        onClick={handleClose}
                        className={`ml-3 ${config.textColor} hover:opacity-75 transition-opacity`}
                    >
                        <X className="h-4 w-4" />
                    </button>
                </div>
            </div>
        </div>
    );
};

// Hook pour gérer les toasts
export const useToast = () => {
    const [toasts, setToasts] = useState([]);

    const addToast = (type, message, duration = 5000) => {
        const id = Date.now();
        setToasts(prev => [...prev, { id, type, message, duration }]);
    };

    const removeToast = (id) => {
        setToasts(prev => prev.filter(toast => toast.id !== id));
    };

    const showSuccess = (message, duration) => addToast('success', message, duration);
    const showError = (message, duration) => addToast('error', message, duration);
    const showInfo = (message, duration) => addToast('info', message, duration);

    const ToastContainer = () => (
        <div className="fixed top-4 right-4 z-50 space-y-2">
            {toasts.map(toast => (
                <Toast
                    key={toast.id}
                    type={toast.type}
                    message={toast.message}
                    duration={toast.duration}
                    onClose={() => removeToast(toast.id)}
                />
            ))}
        </div>
    );

    return {
        showSuccess,
        showError,
        showInfo,
        ToastContainer
    };
};

export default Toast;